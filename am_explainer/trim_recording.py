import argparse
# Script to trim an rtl-sdr recording file given a duration and starting offset in seconds.
# Usage: python trim_recording.py <input> <output> <sample rate> <start offset> <duration>
# Example: python trim_recording.py twr.bin twr_rx.bin 1.06e6 12 5

def trim_sdr_recording(file_path, output_path, sample_rate, start_offset_sec, duration_sec):
    num_samples_to_skip = int(sample_rate * start_offset_sec)
    num_samples_to_keep = int(sample_rate * duration_sec)

    bytes_per_sample = 2  # Assuming each I/Q pair is 2 bytes
    start_byte = num_samples_to_skip * bytes_per_sample
    num_bytes_to_read = num_samples_to_keep * bytes_per_sample

    with open(file_path, 'rb') as infile:
        infile.seek(start_byte)
        segment = infile.read(num_bytes_to_read)

    with open(output_path, 'wb') as outfile:
        outfile.write(segment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trim a raw SDR recording.")
    parser.add_argument('input_file', type=str, help='Path to the input raw IQ file.')
    parser.add_argument('output_file', type=str, help='Path to save the trimmed output file.')
    parser.add_argument('sample_rate', type=float, help='Sample rate of the recording in samples per second.')
    parser.add_argument('start_offset_sec', type=float, help='Starting offset in seconds.')
    parser.add_argument('duration_sec', type=float, help='Desired duration in seconds.')

    args = parser.parse_args()

    trim_sdr_recording(args.input_file, args.output_file, args.sample_rate, args.start_offset_sec, args.duration_sec)
