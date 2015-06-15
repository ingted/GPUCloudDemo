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

let simulatePI rng seed numPointsLog numPointsPerStreamLog numStreamsPerTaskLog =
    let numStreamsLog = numPointsLog - numPointsPerStreamLog
    let numTasksLog = numStreamsLog - numStreamsPerTaskLog
    let numPoints = 1L <<< numPointsLog
    let numPointsPerStream = 1 <<< numPointsPerStreamLog
    let numStreamsPerTask = 1 <<< numStreamsPerTaskLog
    let numStreams = 1 <<< numStreamsLog
    let numTasks = 1 <<< numTasksLog

    printfn "numPoints              : 1 <<< %d = %d" numPointsLog numPoints
    printfn "numPointsPerStream     : 1 <<< %d = %d" numPointsPerStreamLog numPointsPerStream
    printfn "numStreamsPerTask      : 1 <<< %d = %d" numStreamsPerTaskLog numStreamsPerTask
    printfn "numStreams             : 1 <<< %d = %d" numStreamsLog numStreams
    printfn "numTasks               : 1 <<< %d = %d" numTasksLog numTasks

    let pi =
        [| 0..numTasks - 1 |]
        |> Array.map (fun taskId -> 
            { Seed = seed
              NumStreams = numStreams
              NumPoints = numPointsPerStream
              StartStreamId = taskId * numStreamsPerTask
              StopStreamId = (taskId + 1) * numStreamsPerTask - 1
              GetRandom = rng } )
        |> CloudFlow.ofArray
        |> CloudFlow.map calcPI
        |> CloudFlow.toArray
        |> cluster.Run
        |> Array.choose id
        |> Array.average

    printfn "PI                     : %.12f" pi

let rngXorShift7 seed numStreams numDimensions = Rng.XorShift7.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>
let rngMrg32k3a  seed numStreams numDimensions = Rng.Mrg32k3a.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>

let numPointsLog = 32
// 23 results around 8 million points, that is roughly a good choice
let numPointsPerStreamLog = 23
// 5 results 32 streams per task, running this configure results tasks=64, which is good if your gpu
// workers is less than 64 (I think mostly it is)
let numStreamsPerTaskLog = 5
let seed = 33u

simulatePI rngMrg32k3a seed numPointsLog numPointsPerStreamLog numStreamsPerTaskLog

