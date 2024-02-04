from sudio.types import LibSampleFormat, SampleFormat



# Mapping from SampleFormat to LibSampleFormat
SampleFormatToLib = {
    SampleFormat.formatInt16: LibSampleFormat.SIGNED16,
    SampleFormat.formatFloat32: LibSampleFormat.FLOAT32,
    SampleFormat.formatInt32: LibSampleFormat.SIGNED32,
    SampleFormat.formatUInt8: LibSampleFormat.UNSIGNED8,
    SampleFormat.formatInt8: LibSampleFormat.UNSIGNED8,
    SampleFormat.formatUnknown: LibSampleFormat.UNKNOWN,
}

# Mapping from LibSampleFormat to SampleFormat
LibToSampleFormat = {value: key for key, value in SampleFormatToLib.items()}

# Mapping from SampleFormat enum values to corresponding LibSampleFormat values
SampleFormatEnumToLib = {key.value: value for key, value in SampleFormatToLib.items()}

# Mapping from LibSampleFormat enum values to corresponding SampleFormat values
LibSampleFormatEnumToSample = {key.value: key for key, _ in SampleFormatToLib.items()}



