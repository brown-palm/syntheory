use rustysynth::{Synthesizer, SoundFont, SynthesizerSettings, MidiFile, MidiFileSequencer};
use hound::{WavWriter, SampleFormat};

use std::fs::File;
use std::env;
use std::sync::Arc;

const SAMPLE_RATE: i32 = 44_100;

fn save_wave_file(left_channel: &[f32], right_channel: &[f32], sample_rate: u32, file_path: &str) -> Result<(), hound::Error> {
    assert_eq!(left_channel.len(), right_channel.len());
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32, // can be 16 bit too
        sample_format: SampleFormat::Float, 
    };

    let mut writer = WavWriter::create(file_path, spec)?;

    for (left, right) in left_channel.iter().zip(right_channel.iter()) {
        writer.write_sample(*left)?;
        writer.write_sample(*right)?;
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <soundfont> <midi> <out_path>", args[0]);
        std::process::exit(1);
    }

    let soundfont_file = &args[1];
    let midi_file = &args[2];
    let out_path = &args[3];

    println!("soundfont file: {}", soundfont_file);
    println!("MIDI file: {}", midi_file);
    println!("output file: {}", out_path);

    let mut sf2 = File::open(soundfont_file).unwrap();
    let sound_font = Arc::new(SoundFont::new(&mut sf2).unwrap());

    let mut mid = File::open(midi_file).unwrap();
    let midi_file = Arc::new(MidiFile::new(&mut mid).unwrap());

    let settings = SynthesizerSettings::new(SAMPLE_RATE);
    let synthesizer = Synthesizer::new(&sound_font, &settings).unwrap();
    let mut sequencer = MidiFileSequencer::new(synthesizer);

    sequencer.play(&midi_file, false);

    let sample_count = (settings.sample_rate as f64 * midi_file.get_length()) as usize;
    let mut left: Vec<f32> = vec![0_f32; sample_count];
    let mut right: Vec<f32> = vec![0_f32; sample_count];

    sequencer.render(&mut left[..], &mut right[..]);

    let u32_sample_rate: u32 = SAMPLE_RATE as u32;
    if let Err(err) = save_wave_file(&left, &right, u32_sample_rate, out_path) {
        eprintln!("error: {}", err);
    } else {
        println!("wav file saved.");
    }
}
