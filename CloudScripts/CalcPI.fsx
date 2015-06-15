#load "Common.fsx"
#r "../MyApp.FS/bin/Release/MyApp.FS.exe"

open System
open System.IO
open System.Threading
open MBrace
open MBrace.Azure
open MBrace.Azure.Client
open MBrace.Azure.Runtime
open MBrace.Workflows
open MBrace.Flow
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open MyApp.FS.CalcPI
open CloudScripts

let cluster = Runtime.GetHandle(config)
cluster.ClearAllProcesses()

//let numCloudWorkers = (cluster.GetWorkers(showInactive = false) |> Array.ofSeq).Length

let simulatePI rng seed numPointsLog numPointsPerStreamLog =
    let numStreamsLog = numPointsLog - numPointsPerStreamLog
    let numPoints = 1 <<< numPointsLog
    let numPointsPerStream = 1 <<< numPointsPerStreamLog
    let numStreams = 1 <<< numStreamsLog

    printfn "numPoints              : 1 <<< %d = %d" numPointsLog numPoints
    printfn "numPointsPerStream     : 1 <<< %d = %d" numPointsPerStreamLog numPointsPerStream
    printfn "numStreams             : 1 <<< %d = %d" numStreamsLog numStreams

    let pi =
        [| 0..numStreams - 1 |]
        |> Array.map (fun streamId -> 
            { Seed = seed
              NumStreams = numStreams
              NumPoints = numPointsPerStream
              StreamId = streamId
              GetRandom = rng } )
        |> CloudFlow.ofArray
        |> CloudFlow.map calcPI
        |> CloudFlow.toArray
        |> cluster.Run
        |> Array.choose id
        |> Array.average

    printfn "PI                     : %f" pi

let rngXorShift7 seed numStreams numDimensions = Rng.XorShift7.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>
let rngMrg32k3a  seed numStreams numDimensions = Rng.Mrg32k3a.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>

let numPointsLog = 32
let numPointsPerStreamLog = 20
let seed = 42u

simulatePI rngXorShift7 seed numPointsLog numPointsPerStreamLog

