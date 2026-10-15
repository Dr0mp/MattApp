"""One owned, windowless SAM runtime; reused for previews and reaped on cancel/close."""
import subprocess,tempfile,threading,json,os,time


class OwnedJob:
    def __init__(self,process):
        self.handle=None
        if os.name!='nt':return
        import ctypes
        from ctypes import wintypes
        class BASIC(ctypes.Structure):
            _fields_=[('ProcessTime',ctypes.c_longlong),('JobTime',ctypes.c_longlong),('Flags',wintypes.DWORD),('Min',ctypes.c_size_t),('Max',ctypes.c_size_t),('Count',wintypes.DWORD),('Affinity',ctypes.c_size_t),('Priority',wintypes.DWORD),('Scheduling',wintypes.DWORD)]
        class IO(ctypes.Structure):_fields_=[('Value',ctypes.c_ulonglong*6)]
        class EXTENDED(ctypes.Structure):_fields_=[('Basic',BASIC),('IO',IO),('ProcessLimit',ctypes.c_size_t),('JobLimit',ctypes.c_size_t),('PeakProcess',ctypes.c_size_t),('PeakJob',ctypes.c_size_t)]
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];api.CreateJobObjectW.restype=wintypes.HANDLE
        api.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE];api.CloseHandle.argtypes=[wintypes.HANDLE]
        api.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT];api.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
        handle=api.CreateJobObjectW(None,None);limits=EXTENDED();limits.Basic.Flags=0x2000
        if not handle or not api.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not api.AssignProcessToJobObject(handle,int(process._handle)):
            if handle:api.CloseHandle(handle)
            raise OSError('Could not bind the model runtime to the app lifetime')
        self.handle=handle;self.api=api
        # Popen creates the initial thread suspended. Assign the job before the
        # venv launcher or model can spawn children, then resume that thread.
        class THREAD(ctypes.Structure):_fields_=[('Size',wintypes.DWORD),('Usage',wintypes.DWORD),('ID',wintypes.DWORD),('Owner',wintypes.DWORD),('Priority',ctypes.c_long),('Delta',ctypes.c_long),('Flags',wintypes.DWORD)]
        api.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];api.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
        api.Thread32First.argtypes=[wintypes.HANDLE,ctypes.POINTER(THREAD)];api.Thread32Next.argtypes=[wintypes.HANDLE,ctypes.POINTER(THREAD)]
        api.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenThread.restype=wintypes.HANDLE;api.ResumeThread.argtypes=[wintypes.HANDLE];api.ResumeThread.restype=wintypes.DWORD
        snapshot=api.CreateToolhelp32Snapshot(4,0);entry=THREAD();entry.Size=ctypes.sizeof(entry);resumed=False
        try:
            found=api.Thread32First(snapshot,ctypes.byref(entry))
            while found:
                if entry.Owner==process.pid:
                    thread=api.OpenThread(2,False,entry.ID)
                    if thread:
                        resumed=api.ResumeThread(thread)!=0xffffffff;api.CloseHandle(thread)
                    break
                found=api.Thread32Next(snapshot,ctypes.byref(entry))
        finally:api.CloseHandle(snapshot)
        if not resumed:self.close();raise OSError('Could not resume the owned runtime')
    def close(self):
        if self.handle:
            import ctypes
            from ctypes import wintypes
            class ACCOUNTING(ctypes.Structure):_fields_=[('Times',ctypes.c_longlong*4),('Faults',wintypes.DWORD),('Total',wintypes.DWORD),('Active',wintypes.DWORD),('Terminated',wintypes.DWORD)]
            self.api.TerminateJobObject(self.handle,1)
            deadline=time.monotonic()+5;info=ACCOUNTING()
            while time.monotonic()<deadline and self.api.QueryInformationJobObject(self.handle,1,ctypes.byref(info),ctypes.sizeof(info),None) and info.Active:time.sleep(.01)
            self.api.CloseHandle(self.handle);self.handle=None


class Session:
    def __init__(self):self.process=None;self.job=None;self.errors=None;self.lock=threading.RLock();self.identity=None
    def stop(self):
        with self.lock:
            if self.job:self.job.close();self.job=None
            if self.process:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                if self.process.stdin:self.process.stdin.close()
            self.process=None;self.identity=None
            if self.errors:self.errors.close();self.errors=None
    def run(self,python,runner,request,token,collect):
        with self.lock:
            identity=(str(python),str(runner))
            if not self.process or self.process.poll() is not None or self.identity!=identity:
                self.stop();self.errors=tempfile.TemporaryFile()
                self.process=subprocess.Popen([str(python),str(runner),'--server'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.errors,creationflags=(subprocess.CREATE_NO_WINDOW|0x4) if os.name=='nt' else 0)
                try:self.job=OwnedJob(self.process)
                except Exception:self.stop();raise
                self.identity=identity
            status=request.parent/'server-status.json'
            try:
                token.check();self.process.stdin.write((str(request)+'\n').encode('utf-8'));self.process.stdin.flush()
                while True:
                    token.check();collect()
                    if status.exists():
                        data=json.loads(status.read_text())
                        if data.get('error'):raise RuntimeError(data['error'])
                        return
                    if self.process.poll() is not None:
                        self.errors.seek(0);raise RuntimeError(self.errors.read().decode('utf-8',errors='replace') or 'Model runtime exited unexpectedly')
                    token.wait(.05)
            except BaseException:
                self.stop();raise

SESSION=Session()
def stop_session():SESSION.stop()
