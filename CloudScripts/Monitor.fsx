#load "Common.fsx"

open System
open System.IO
open System.Threading
open MBrace
open MBrace.Azure
open MBrace.Azure.Client
open MBrace.Azure.Runtime
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open CloudScripts

let cluster = Runtime.GetHandle(config)

// run this from cmd window to monitor the status
while true do
    cluster.ShowWorkers()
    cluster.ShowProcesses()
    Thread.Sleep(5000)
