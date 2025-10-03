//
// Created by namnt55 on 9/26/25.
//

#ifndef VOICEBIO_WAV_H
#define VOICEBIO_WAV_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

struct WavHeader {
  char riff[4];
  unsigned int size;
  char wav[4];
  char fmt[4];
  unsigned int fmt_size;
  uint16_t format;
  uint16_t channels;
  unsigned int sample_rate;
  unsigned int bytes_per_second;
  uint16_t block_size;
  uint16_t bit;
  char data[4];
  unsigned int data_size;
};

class WavReader {
public:
  WavReader();
  explicit WavReader(const std::string &file_name);
  ~WavReader();

  bool open(const std::string &file_name);
  int num_channel() const;
  int sample_rate() const;
  int bits_per_sample() const;
  int num_samples() const;
  const float *data() const;

private:
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
  int num_samples_;
  float *data_;
};

class WavWriter {
public:
  WavWriter(const float *data, int num_samples, int num_channel,
            int sample_rate, int bits_per_sample);
  void write(const std::string &file_name);

private:
  const float *data_;
  int num_samples_;
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
};

#endif //VOICEBIO_WAV_H
