### How to Make Custom Datasets

We aspire to make it easy for others to use our codebase to design custom datasets. Here we document the basics of what that entails.

Dataset creation requires us to define two functions:
- the `row_iterator`
- the `row_processor`

These are arguments to the constructor of the `DatasetWriter` class. Examples of this class being used to define a dataset may be found in all the `main` functions in the `dataset/synthetic/` folder. 

Let's look at an example using the Chord Progressions dataset (see `dataset/synthetic/chord_progressions.py`). 

The main function begins with:
```python
dataset_name = "chord_progressions"
dataset_writer = DatasetWriter(
    dataset_name=dataset_name,
    save_to_parent_directory=OUTPUT_DIR,
    # ----- row processor and row iterator ----
    row_iterator=get_row_iterator(
        progressions=PROGRESSIONS,
        keys=get_all_keys(),
        instruments=get_instruments(
            ignore_atonal=True,
            ignore_polyphonic=True,
            ignore_highly_articulate=True,
            take_only_first_category=False,
        ),
    ),
    row_processor=row_processor,
    # -----------------------------------------
    max_processes=8,
)
```

#### `row_iterator` function

`row_iterator` is a function that returns an iterator (specifically the type `Iterator[DatasetRowDescription]`) over all the rows for the dataset. 

There is nothing special about the `DatasetRowDescription` type. It is just an alias for:
```python
DatasetRowDescription = Tuple[int, Dict[str, Any]]
```

This just means that the `row_iterator` function should return a tuple containing the index of the row along with all the information we need to uniquely describe it. For example:
```
(10, {
    "chord": "C major",
    "inversion": "first"
})
```
>Just an illustration. Check out the `get_row_iterator` functions in each of the dataset files to see more realistic examples.

#### `row_processor` function

`row_processor` is a function that takes as input the `Path` object containing the dataset itself and the `DatasetRowDescription` object that is returned by calling `row_iterator`. The `dataset_path` parameter is given when `row_processor` is called from within the `DatasetWriter` class. 

There is no need to call this function explicitly, passing the _function_ as an argument in the constructor of `DatasetWriter` is sufficient. The `row_processor` function is called for each of the rows. This function can have side-effects on disk (e.g. writing several files). It will also be called in a multi-processing context (unless `max_processes` is set to `1` in the constructor of `DatasetWriter`). Therefore, we must make any paths written by this function unique per row. Otherwise, a file may be overwritten. 

`row_processor` may return several rows. This might be a bit confusing, but it is to allow several rows in the final dataset to be derived from the same input information. As long as each of these rows have unique sample names / file artifacts, it is valid for `row_processor` to return more than one row. Alternatively, one may prefer to instead keep a 1-to-1 relationship between an output of `row_iterator` and `row_processor` by creating any variations in `row_iterator` and passing the necessary information to create those variations in `row_processor`. Either is fine. 

To be more specific about what `row_processor` must return, we need a list of rows, where each row is a tuple of that row's index and a JSON-like object that describes the columns and their values for that row. Here is an example:
```python
return [
    (
        0, 
        {
            "chord": "C major",
            "inversion": "first",
            "file_location": "./dataset/my_chord.wav"
        }
    )
]
```

A common pattern when writing the `row_processor` function is to do something like this:
```python
def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    # get the row index and information required to create the row
    row_idx, row_info = row

    # example of pulling some information we need
    progression = row_info["progression"]
    # ... 
    # ... do some sample creation / file writing
    # ... 

    # this is a helpful function for checking if your function is writing blank audio files
    is_silent = is_wave_silent(audio_file_path)

    # record this row in the csv
    return [
        (
            # returning the row index, it can be the same as the input row. Ensure that this
            # value is unique for all rows
            row_idx,
            {
                # add some information about the row
                "progression": progression,
                "chord_progression": chord_progression_label,

                # the `.relative_to` function call is a neat trick to help the dataset folders 
                # be more robust to moving them to a new location on disk
                "audio_file_path": str(audio_file_path.relative_to(dataset_path)),
                
                # record whether this audio file is silent or not
                "is_silent": is_silent,
            },
        )
    ]
```

#### Small Example

So let's say we want to add a new chord progression to the Chord Progressions dataset. Given what we know now, we would need to somehow change the `row_iterator` function to add another progression class. In this case, the `row_iterator` function takes an array of progression definitions named `PROGRESSIONS`. Let's say we want to add something in the Dorian mode: i-VII-ii-III. We can add this to the `PROGRESSIONS` array:
```python
PROGRESSIONS = (
    # tuple of mode, scale_degrees
    ("ionian", (1, 4, 5, 1)),
    # ...
    ("aeolian", (4, 7, 1, 1)),
    ("aeolian", (7, 6, 7, 1)),
    # our new progression
    ("dorian", (1, 7, 2, 3)),
)
```

Now, the dataset will contain our additional chord progression when we generate it.

What if we now want to support 7th chords in our progressions? So far, we have only triads. Examine the `get_progression_midi_notes` function in `dataset/synthetic/chord_progressions.py`. It looks something like this:
```python
def get_progression_midi_notes(
    key_pitch_class: int, mode: str, progression_degrees: Tuple[int, ...]
):
    scale = get_scale(key_pitch_class, mode)
    midi_notes = []
    for chord in progression_degrees:
        chord_midi_notes = get_chord(
            scale,
            mode=mode,
            root=chord,
            inversion=None,
            # triads only for now
            chord_type=5,
            extensions=[],
            borrowed=None,
        )[1]
        midi_notes.append(chord_midi_notes)
    return midi_notes
```

Tracing the code a bit more to what calls this function, we can see that `get_progression_midi_notes` basically takes a single element in `PROGRESSIONS` and turns it into a list of the MIDI notes that comprise that chord. It takes as input, through the `progression_degrees` argument a tuple of integers. We saw this earlier when we added the new progression; the degrees were: `(1, 7, 2, 3)`

To support 7th chords, we should modify the way we represent progressions in the `PROGRESSIONS` object. Instead of having a tuple of the chord degrees as integers, we might now prefer a tuple of tuples:
```
(1, 7, 2, 3) --> ((1, 5), (7, 7), (2, 5), (3, 5))
``` 

Where now, each element is of the form `(chord degree, chord type)`. In the above, we changed the VII into a VIImaj7. We now need to change the code to accomodate this change:
```python
def get_progression_midi_notes(
    key_pitch_class: int, mode: str, progression_degrees: Tuple[Tuple[int, int], ...]
):
    scale = get_scale(key_pitch_class, mode)
    midi_notes = []
    for chord in progression_degrees:
        chord_deg, chord_type = chord
        chord_midi_notes = get_chord(
            scale,
            mode=mode,
            root=chord_deg,
            inversion=None,
            chord_type=chord_type,
            extensions=[],
            borrowed=None,
        )[1]
        midi_notes.append(chord_midi_notes)
    return midi_notes
```

That is nice, but we still only support root position chords with no inversions. Also, these progression only contain four chords. If you can imagine how to implement it, it can be done! The hope is that some of the MIDI manipulation and music theory related functions will be useful (see `dataset/music/...`). By using these functions, it would hopefully be easier to define new constructs and concepts in terms of music theory instead of implementing this from scratch.

This small example is not exhaustive. We encourage readers and users of the codebase to expand our very basic setup and concepts with their own ideas and variations. 