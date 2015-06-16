using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading;
using Alea.CUDA;
using Alea.CUDA.IL;
using Alea.CUDA.Unbound;
using Microsoft.FSharp.Core;

namespace MyApp.CS
{
    [Serializable]
    public class CalcPIParam
    {
        public uint Seed { get; set; }
        public int NumStreams { get; set; }
        public int NumPoints { get; set; }
        public int StartStreamId { get; set; }
        public int StopStreamId { get; set; }
        public Func<uint, int, int, Alea.CUDA.Unbound.Rng.IRandom<double>> GetRandom { get; set; } 
    }

    public class CalcPI
    {
        [AOTCompile]
        static public void KernelCountInside(deviceptr<double> pointsX, deviceptr<double> pointsY, int numPoints,
            deviceptr<int> numPointsInside)
        {
            var start = blockIdx.x*blockDim.x + threadIdx.x;
            var stride = gridDim.x*blockDim.x;
            for (var i = start; i < numPoints; i += stride)
            {
                var x = pointsX[i];
                var y = pointsY[i];
                numPointsInside[i] = Math.Sqrt(x*x + y*y) <= 1.0 ? 1 : 0;
            }
        }

        static private bool CanDoGPUCalc
        {
            get
            {
                try
                {
                    var x = Device.Default;
                    return true;
                }
                catch (Exception)
                {
                    return false;
                }
            }
        }

        static private double _Calc(Worker worker, CalcPIParam param)
        {
            var seed = param.Seed;
            var numStreams = param.NumStreams;
            const int numDimentions = 2;
            var numPoints = param.NumPoints;
            var numSMs = worker.Device.Attributes.MULTIPROCESSOR_COUNT;
            var startStreamId = param.StartStreamId;
            var stopStreamId = param.StopStreamId;

            var random = param.GetRandom(seed, numStreams, numDimentions);
            using (var reduce = DeviceSumModuleI32.Default.Create(numPoints))
            using (var points = random.AllocCUDAStreamBuffer(numPoints))
            using (var numPointsInside = worker.Malloc<int>(numPoints))
            {
                var pointsX = points.Ptr;
                var pointsY = points.Ptr + numPoints;
                var lp = new LaunchParam(numSMs * 8, 256);
                var countStream = stopStreamId + 1 - startStreamId;

                var pi =
                    Enumerable.Range(startStreamId, countStream).Select(streamId =>
                    {
                        random.Fill(streamId, numPoints, points);
                        worker.Launch(KernelCountInside, lp, pointsX, pointsY, numPoints, numPointsInside.Ptr);
                        var numPointsInsideH = reduce.Reduce(numPointsInside.Ptr, numPoints);
                        return 4.0 * (double)numPointsInsideH / (double)numPoints / (double)countStream;
                    }).Aggregate((a, b) => a + b);

                Console.WriteLine("Streams({0}-{1}/{2}) Points({3}) : {4}", 
                    startStreamId + 1,
                    stopStreamId + 1,
                    numStreams,
                    numPoints,
                    pi);

                return pi;
            }
        }

        static public FSharpOption<double> Calc(CalcPIParam param)
        {
            if (CanDoGPUCalc)
            {
                var worker = Worker.Default;
                var pi = worker.EvalFunc(() => _Calc(worker, param));
                return FSharpOption<double>.Some(pi);
            }
            return FSharpOption<double>.None;
        }
    }
}