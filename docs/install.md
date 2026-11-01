# Install

One or the other:

```
uv add screws                 # the mathematics only; numpy and SymPy are the dependencies
uv add "screws[coppelia]"     # the same, plus the CoppeliaSim ZMQ remote API client
```

Extras for the bridge: `screws[plot]` (matplotlib, for `Log.plot`) and `screws[video]`
(imageio and ffmpeg, for `Recorder`). `uv add "screws[coppelia,plot,video]"` takes all three.
