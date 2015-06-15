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
open MyApp.FS
open CloudScripts

let cluster = Runtime.GetHandle(config)
cluster.ClearAllProcesses()

// A function to create a list of calcPI task parameters. We randomly select the seed and rng.
let createParams (numPoints:int) (numStreamsPerSM:int) (numRuns:int) : CalcPI.CalcParam[] =
    let rng = Random()
    Array.init numRuns (fun taskId ->
        let seed = rng.Next() |> uint32
        let getRandomXorshift7 numStreams numDimensions = Rng.XorShift7.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>
        let getRandomMrg32k3a  numStreams numDimensions = Rng.Mrg32k3a.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>
        let getRandom =
            match rng.Next(2) with
            | 0 -> getRandomXorshift7
            | _ -> getRandomMrg32k3a
        { TaskId = taskId; NumPoints = numPoints; NumStreamsPerSM = numStreamsPerSM; GetRandom = getRandom } )

let oneMillion = 1000000
let numCloudWorkers = (cluster.GetWorkers(showInactive = false) |> Array.ofSeq).Length

let numSamples = numCloudWorkers*10*oneMillion
let numTasks = numCloudWorkers*10
let numStreamsPerSM = 10
let numRuns = numCloudWorkers * 100

// This is the cloud workflow, we have a big calculation (numRuns task, each task will generate many 
// random streams, and approximate PI with these random numbers). CloudFlow.map will map these tasks
// to available cloud workers. Because not all cloud workers have a GPU, we return float option.
// We choose those results which return some number and take the mean.
let pi = 
    createParams numSamples numStreamsPerSM numRuns
    |> CloudFlow.ofArray
    |> CloudFlow.map CalcPI.calcPI
    |> CloudFlow.toArray
    |> cluster.Run
    |> Array.choose id
    |> Array.average

printfn "PI = %A" pi
