Option Explicit
Dim shell, fs, root, cfg, line, runtimeHome, pythonExe, command
Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
root = fs.GetParentFolderName(fs.GetParentFolderName(WScript.ScriptFullName))

pythonExe = ""

If fs.FileExists(root & "\.venv\Scripts\pythonw.exe") Then
    pythonExe = root & "\.venv\Scripts\pythonw.exe"
ElseIf fs.FileExists(root & "\.venv\pyvenv.cfg") Then
    Set cfg = fs.OpenTextFile(root & "\.venv\pyvenv.cfg", 1)
    runtimeHome = ""
    Do Until cfg.AtEndOfStream
        line = cfg.ReadLine
        If Left(line, 7) = "home = " Then runtimeHome = Mid(line, 8)
    Loop
    cfg.Close
    If fs.FileExists(runtimeHome & "\pythonw.exe") Then
        pythonExe = runtimeHome & "\pythonw.exe"
    End If
End If

If pythonExe = "" Then
    Dim answer
    answer = MsgBox("Matte App environment (.venv) was not found." & vbCrLf & vbCrLf & "Would you like to run the installer now?", 36, "Matte App Setup")
    If answer = 6 Then
        shell.CurrentDirectory = root
        shell.Run Chr(34) & root & "\scripts\install.bat" & Chr(34), 1, True
    End If
    WScript.Quit 1
End If

shell.CurrentDirectory = root
shell.Environment("PROCESS")("__PYVENV_LAUNCHER__") = root & "\.venv\Scripts\pythonw.exe"
command = Chr(34) & pythonExe & Chr(34) & " " & Chr(34) & root & "\scripts\launch_app.pyw" & Chr(34)
shell.Run command, 1, False
