﻿#load "Config.fsx"
#r "../packages/Streams/lib/net45/Streams.Core.dll"
#r "../packages/MBrace.Flow/lib/net45/MBrace.Flow.dll"
#r "../packages/Alea.IL/lib/net40/Alea.IL.dll"
#r "../packages/Alea.CUDA/lib/net40/Alea.CUDA.dll"
#r "../packages/Alea.CUDA.IL/lib/net40/Alea.CUDA.IL.dll"
#r "../packages/Alea.CUDA.Unbound/lib/net40/Alea.CUDA.Unbound.dll"
#r "../MyApp.FS/bin/Release/MyApp.FS.exe"

open MBrace.Azure.Client
open MBrace.Flow
open Alea.CUDA.Unbound
open MyApp.FS.CalcPI

let cluster = Runtime.GetHandle(Config.config)

let simulatePI rng seed numPointsPerStream numStreamsPerTask numTasks =
    printfn "numPointsPerStream     : %d" numPointsPerStream
    printfn "numStreamsPerTask      : %d" numStreamsPerTask
    printfn "numTasks               : %d" numTasks

    let numStreams = numStreamsPerTask * numTasks
    let numTotalPoints = (int64 numPointsPerStream) * (int64 numStreamsPerTask) * (int64 numTasks)

    printfn "numStreams             : %d" numStreams
    printfn "numTotalPoints         : %d" numTotalPoints

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

let oneMillion = 1000000
let numPointsPerStream = 10 * oneMillion
let numStreamsPerTask = 32
let numTasks = 64

simulatePI rngXorShift7 42u numPointsPerStream numStreamsPerTask numTasks

