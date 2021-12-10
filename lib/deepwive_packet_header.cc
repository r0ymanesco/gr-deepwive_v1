/* -*- c++ -*- */
/*
 * Copyright 2021 gr-deepwive_v1 author.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include <deepwive_v1/deepwive_packet_header.h>
#include <gnuradio/digital/lfsr.h>
#include <iostream>
// #include <gnuradio/digital/packet_header_ofdm.h>

namespace gr {
namespace deepwive_v1 {

int _get_header_len_from_occupied_carriers(
    const std::vector<std::vector<int>>& occupied_carriers, int n_syms)
{
    int header_len = 0;
    for (int i = 0; i < n_syms; i++) {
        header_len += occupied_carriers[i].size();
    }

    return header_len;
}

deepwive_packet_header::sptr
deepwive_packet_header::make(
  const std::vector<std::vector<int>>& occupied_carriers,
  int n_syms,
  const std::string &len_tag_key,
  const std::string &frame_len_tag_key,
  const std::string &num_tag_key,
  int bits_per_header_sym,
  int bits_per_payload_sym,
  bool scramble_header)
{
  return deepwive_packet_header::sptr(
    new deepwive_packet_header(
      occupied_carriers,
      n_syms,
      len_tag_key,
      frame_len_tag_key,
      num_tag_key,
      bits_per_header_sym,
      bits_per_payload_sym,
      scramble_header)
  );
}

deepwive_packet_header::deepwive_packet_header(
    const std::vector<std::vector<int>> &occupied_carriers,
    int n_syms,
    const std::string &len_tag_key,
    const std::string &frame_len_tag_key,
    const std::string &num_tag_key,
    int bits_per_header_sym,
    int bits_per_payload_sym,
    bool scramble_header)
    : packet_header_default(
          _get_header_len_from_occupied_carriers(occupied_carriers, n_syms),
          len_tag_key,
          num_tag_key,
          bits_per_header_sym),
      Md_header_len(_get_header_len_from_occupied_carriers(occupied_carriers, n_syms)),
      Md_len_tag_key(pmt::string_to_symbol(len_tag_key)),
      Md_num_tag_key(num_tag_key.empty() ? pmt::PMT_NIL : pmt::string_to_symbol(num_tag_key)),
      Md_bits_per_byte(bits_per_header_sym),
      Md_header_number(0),
      d_frame_len_tag_key(pmt::string_to_symbol(frame_len_tag_key)),
      d_occupied_carriers(occupied_carriers),
      d_bits_per_payload_sym(bits_per_payload_sym),
      d_scramble_mask(d_header_len, 0)
{
    // Init scrambler mask
    if (scramble_header) {
        // These are just random values which already have OK PAPR:
        gr::digital::lfsr shift_reg(0x8a, 0x6f, 7);
        for (int i = 0; i < d_header_len; i++) {
            for (int k = 0; k < bits_per_header_sym; k++) {
                d_scramble_mask[i] ^= shift_reg.next_bit() << k;
            }
        }
    }
}

deepwive_packet_header::~deepwive_packet_header() {}

  void deepwive_packet_header::insert_into_header_buffer(
    unsigned char *out,
    int &currentOffset,
    unsigned value_to_insert,
    int number_of_bits_to_copy)
{
  //using namespace std;
  //cout << "Number to insert " << value_to_insert << endl;
  for (int i = 0; i < number_of_bits_to_copy && currentOffset < Md_header_len; i += Md_bits_per_byte, currentOffset++)
  {
    out[currentOffset] = (unsigned char)((value_to_insert >> i) & d_mask);
  }
}

  unsigned deepwive_packet_header::extract_from_header_buffer(
    std::vector<unsigned char> &in,
    int &currentOffset,
    int size_of_field)
{
  unsigned result = 0;

  for (int i = 0; i < size_of_field && currentOffset < Md_header_len; i += Md_bits_per_byte, currentOffset++)
  {
    result |= (((int)in[currentOffset]) & d_mask) << i;
  }

  return result;
}

  bool deepwive_packet_header::header_formatter(
    long packet_len,
    unsigned char *out,
    const std::vector<tag_t> &tags)
{

  unsigned frame_idx = 0;
  unsigned packet_idx = 0;

  for (size_t i = 0; i < tags.size(); i++)
  {
    if (pmt::equal(tags[i].key, pmt::intern("frame_idx")))
    {
      frame_idx = static_cast<unsigned int>(pmt::to_long(tags[i].value));
    }

    if (pmt::equal(tags[i].key, pmt::intern("packet_idx")))
    {
      packet_idx = static_cast<unsigned int>(pmt::to_long(tags[i].value));
    }
  }

  packet_len &= 0x0FF;
  Md_crc_impl.reset();
  Md_crc_impl.process_bytes((void const *)&frame_idx, 2);
  Md_crc_impl.process_bytes((void const *)&packet_idx, 1);
  unsigned char crc = Md_crc_impl();

  memset(out, 0x00, Md_header_len);
  int k = 0;
  for (int i = 0; i < 3; i++) // FIXME this is hard coded to give 48 bits, should find a more flexible solution
  {
    insert_into_header_buffer(out, k, frame_idx, 7);
    insert_into_header_buffer(out, k, packet_idx, 9);
  }
  return true;

  // bool ret_val = packet_header_default::header_formatter(packet_len, out, tags);
  // for (int i = 0; i < d_header_len; i++) {
  //   out[i] ^= d_scramble_mask[i];
  // }
  // return ret_val;
}

  bool deepwive_packet_header::header_parser(const unsigned char *in,
                                             std::vector<tag_t> &tags)
{

  std::vector<unsigned char> in_descrambled(d_header_len, 0);
  for (int i = 0; i < d_header_len; i++)
  {
    in_descrambled[i] = in[i];
  }

  tag_t tagH;

  int k = 0; // Position in "in"

  std::vector<unsigned> header_frame_idx(3);
  std::vector<unsigned> header_packet_idx(3);

  for (int i = 0; i < 3; i++)
  {
    header_frame_idx[i] = extract_from_header_buffer(in_descrambled, k, 7);
    header_packet_idx[i] = extract_from_header_buffer(in_descrambled, k, 9);
    // unsigned header_crc = extract_from_header_buffer(in_descrambled,k,8);
  }

  header_frame_idx[0] = (header_frame_idx[0] & header_frame_idx[1]) | (header_frame_idx[1] & header_frame_idx[2]) | (header_frame_idx[0] & header_frame_idx[2]);
  header_packet_idx[0] = (header_packet_idx[0] & header_packet_idx[1]) | (header_packet_idx[1] & header_packet_idx[2]) | (header_packet_idx[0] & header_packet_idx[2]);

  if (k > Md_header_len)
  {
    return false;
  }

  int packet_len = 96; // FIXME find a more elegant solution; cannot change arguments

  tagH.key = pmt::intern("packet_len");
  tagH.value = pmt::from_long(packet_len);
  tags.push_back(tagH);

  tagH.key = pmt::intern("frame_idx");
  tagH.value = pmt::from_long(header_frame_idx[0]);
  tags.push_back(tagH);

  tagH.key = pmt::intern("packet_idx");
  tagH.value = pmt::from_long(header_packet_idx[0]);
  tags.push_back(tagH);

  // std::cout << "frame_idx " << header_frame_idx[0] << std::endl;
  // std::cout << "packet_idx " << header_packet_idx[0] << std::endl;

  // To figure out how many payload OFDM symbols there are in this frame,
  // we need to go through the carrier allocation and count the number of
  // allocated carriers per OFDM symbol.
  // frame_len == # of payload OFDM symbols in this frame
  int frame_len = 0;
  k = 0; // position in the carrier allocation map
  int symbols_accounted_for = 0;
  while (symbols_accounted_for < packet_len)
  {
    frame_len++;
    symbols_accounted_for += d_occupied_carriers[k].size();
    k = (k + 1) % d_occupied_carriers.size();
  }
  tag_t tag;
  tag.key = d_frame_len_tag_key;
  tag.value = pmt::from_long(frame_len);
  tags.push_back(tag);

  return true;
}

} /* namespace deepwive_v1 */
} /* namespace gr */
