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
    public class CalcPIParam
    {
        public int TaskId { get; set; }
        public int NumPoints { get; set; }
        public int NumStreamsPerSM { get; set; }
        //public Func<int, int, Alea.CUDA.Unbound.Rng.IRandom<double>> GetRandom { get; set; } 
    }

    public class CalcPI
    {
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

        static public bool CanDoGPUCalc
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

        static public FSharpOption<double> Calc(CalcPIParam param)
        {
            if (CanDoGPUCalc)
            {
                var worker = Worker.Default;
                var numPoints = param.NumPoints;
                var numStreamsPerSM = param.NumStreamsPerSM;
                var numSMs = worker.Device.Attributes.MULTIPROCESSOR_COUNT;
                var numStreams = numStreamsPerSM*numSMs;
                var numDimensions = 2;

                //var random = param.GetRandom(numStreams, numDimensions);
                var random = (Alea.CUDA.Unbound.Rng.IRandom<double>)Alea.CUDA.Unbound.Rng.Mrg32k3a.CUDA.DefaultUniformRandomModuleF64.Default.Create(
                    numStreams, numDimensions, 42u);

                using (var reduce = DeviceSumModuleI32.Default.Create(numPoints))
                using (var points = random.AllocCUDAStreamBuffer(numPoints))
                using (var numPointsInside = worker.Malloc<int>(numPoints))
                {
                    var pointsX = points.Ptr;
                    var pointsY = points.Ptr + numPoints;
                    var lp = new LaunchParam(numSMs*8, 256);

                    Console.WriteLine("Task #.{0} : Random({1}) Streams({2}) Points({3})", param.TaskId,
                        (random.GetType().Namespace), numStreams, numPoints);

                    var pi = 
                        Enumerable.Range(0, numStreams).Select(streamId =>
                        {
                            random.Fill(streamId, numPoints, points);
                            worker.Launch(KernelCountInside, lp, pointsX, pointsY, numPoints, numPointsInside.Ptr);
                            var numPointsInsideH = reduce.Reduce(numPointsInside.Ptr, numPoints);
                            return 4.0*(double) numPointsInsideH/(double) numPoints/(double)numStreams;
                        }).Aggregate((a, b) => a + b);

                    return FSharpOption<double>.Some(pi);
                }
            }

            return FSharpOption<double>.None;
        }

        static public Alea.CUDA.Unbound.Rng.IRandom<double> GetRandomXorshift7(int numStreams, int numDimensions)
        {
            return Alea.CUDA.Unbound.Rng.XorShift7.CUDA.DefaultUniformRandomModuleF64.Default.Create(numStreams,
                numDimensions, 42u);
        }

        static public void Main()
        {
            var numPoints = 1000000;
            var numStreamsPerSM = 2;

            var param = new CalcPIParam
            {
                TaskId = 0,
                NumPoints = numPoints,
                NumStreamsPerSM = numStreamsPerSM,
                //GetRandom = GetRandomXorshift7
            };

            var pi = Calc(param);
            Console.WriteLine("PI={0}", pi.Value);
        }
    }
}