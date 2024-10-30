### Compiling the Synth

Run: `cargo build`

### Running the Synth

See: `run.sh`, can do something like:

```bash
cargo run -- data/TimGM6mb.sf2 data/715754.mid data/test.wav
```

where the arguments given are
1. soundfont, 
2. midi to play, and 
3. the wav output.

The instruments in the audio depend on the soundfont.

### Acknowledgements

Soundfont is `TimGM6mb.sf2` by Tim Brechbill from here: [LINK](https://timbrechbill.com/saxguru/Timidity.php). 

This code is just a thin wrapper around `rustysynth`, see here: [LINK](https://github.com/sinshu/rustysynth/tree/main)