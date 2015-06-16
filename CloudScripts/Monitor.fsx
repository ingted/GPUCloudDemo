#load "Config.fsx"

open System.Threading
open MBrace.Azure.Client

//run this from cmd window to monitor the status:
//"C:\Program Files (x86)\Microsoft SDKs\F#\3.1\Framework\v4.0\Fsi.exe" Monitor.fsx

let cluster = Runtime.GetHandle(Config.config)

while true do
    cluster.ShowWorkers()
    cluster.ShowProcesses()
    Thread.Sleep(5000)
