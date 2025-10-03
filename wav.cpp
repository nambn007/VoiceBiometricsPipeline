//
// Created by namnt55 on 9/26/25.
//

#include "wav.h"
#include <iostream>

WavReader::WavReader() : data_(nullptr) {

}

WavReader::WavReader(const std::string &file_name) {
  open(file_name);
}

WavReader::~WavReader() {

}

bool WavReader::open(const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "rb");
  if (fp == NULL) {
    std::cerr << "Error in reading file " << file_name << std::endl;
    return false;
  }
  WavHeader header{};
  fread(&header, 1, sizeof(header), fp);

  if (header.fmt_size < 16) {
    std::cerr << "WaveData: expect PCM format data to have fmt chunk of at least size 16." << std::endl;
    return false;
  } else if (header.fmt_size > 16) {
    int offset = 44 - 8 + header.fmt_size - 16;
    fseek(fp, offset, SEEK_SET);
    fread(header.data, 8, sizeof(char), fp);
  }

  while (0 != strncmp(header.data, "data", 4)) {
    fseek(fp, header.data_size, SEEK_CUR);
    fread(header.data, 8, sizeof(char), fp);
  }

  if (header.data_size == 0) {
    int offset = ftell(fp);
    fseek(fp, 0, SEEK_END);
    header.data_size = ftell(fp) - offset;
    fseek(fp, offset, SEEK_SET);
  }

  num_channel_ = header.channels;
  sample_rate_ = header.sample_rate;
  bits_per_sample_ = header.bit;
  int num_data = header.data_size / (bits_per_sample_ / 8);
  data_ = new float[num_data]; // Create 1-dim array
  num_samples_ = num_data / num_channel_;

  std::cout << "num_channel_ " << num_channel_ << std::endl;
  std::cout << "sample_rate_ " << sample_rate_ << std::endl;
  std::cout << "bits_per_sample_ " << bits_per_sample_ << std::endl;
  std::cout << "num_samples_ " << num_samples_ << std::endl;
  std::cout << "data_size " << header.data_size << std::endl;

  switch (bits_per_sample_) {
    case 8: {
      char sample;
      for (int i = 0; i < num_data; ++i) {
        fread(&sample, 1, sizeof(char), fp);
        data_[i] = static_cast<float>(sample) / 32768;
      }
      break;
    }
    case 16: {
      int16_t sample;
      for (int i = 0; i < num_data; ++i) {
        fread(&sample, 1, sizeof(int16_t), fp);
        data_[i] = static_cast<float>(sample) / 32768;
      }
      break;
    }
    case 32:
    {
      if (header.format == 1) //S32
      {
        int sample;
        for (int i = 0; i < num_data; ++i) {
          fread(&sample, 1, sizeof(int), fp);
          data_[i] = static_cast<float>(sample) / 32768;
        }
      }
      else if (header.format == 3) // IEEE-float
      {
        float sample;
        for (int i = 0; i < num_data; ++i) {
          fread(&sample, 1, sizeof(float), fp);
          data_[i] = static_cast<float>(sample);
        }
      }
      else {
        printf("unsupported quantization bits\n");
      }
      break;
    }
    default:
      printf("unsupported quantization bits\n");
      break;
  }

  fclose(fp);
  return true;
}

int WavReader::num_channel() const {
  return num_channel_;
}

int WavReader::sample_rate() const {
  return sample_rate_;
}

int WavReader::bits_per_sample() const {
  return bits_per_sample_;
}

int WavReader::num_samples() const {
  return num_samples_;
}

const float *WavReader::data() const {
  return data_;
}

WavWriter::WavWriter(const float *data, int num_sample_,
                     int num_channel, int sample_rate, int bits_per_sample)
  : data_(data), num_samples_(num_sample_), num_channel_(num_channel),
    sample_rate_(sample_rate), bits_per_sample_(bits_per_sample) {
}

void WavWriter::write(const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "w");
  WavHeader header{};
  char wav_header[44] = {0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57,
                         0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0x00,
                         0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00};
  memcpy(&header, wav_header, sizeof(header));
  header.channels = num_channel_;
  header.bit = bits_per_sample_;
  header.sample_rate = sample_rate_;
  header.data_size = num_samples_ * num_channel_ * (bits_per_sample_ / 8);
  header.size = sizeof(header) - 8 + header.data_size;
  header.bytes_per_second =
      sample_rate_ * num_channel_ * (bits_per_sample_ / 8);
  header.block_size = num_channel_ * (bits_per_sample_ / 8);

  fwrite(&header, 1, sizeof(header), fp);

  for (int i = 0; i < num_samples_; ++i) {
    for (int j = 0; j < num_channel_; ++j) {
      switch (bits_per_sample_) {
        case 8: {
          char sample = static_cast<char>(data_[i * num_channel_ + j]);
          fwrite(&sample, 1, sizeof(sample), fp);
          break;
        }
        case 16: {
          int16_t sample = static_cast<int16_t>(data_[i * num_channel_ + j]);
          fwrite(&sample, 1, sizeof(sample), fp);
          break;
        }
        case 32: {
          int sample = static_cast<int>(data_[i * num_channel_ + j]);
          fwrite(&sample, 1, sizeof(sample), fp);
          break;
        }
      }
    }
  }
  fclose(fp);
}
