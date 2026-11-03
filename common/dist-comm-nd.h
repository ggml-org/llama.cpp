#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// NetworkDirect (Windows RDMA) data path for the distributed-TP ring.
//
// This is a transport substitution only. The ring bootstrap, the op_hdr wire framing and the
// send/recv ordering all stay in dist-comm.cpp; only the bytes of the collective payload move
// over NetworkDirect instead of TCP. The ND connection is negotiated out of band over the ring
// sockets that dist-comm.cpp has already established, and every failure path leaves the caller
// on TCP rather than aborting.
//
// The transport is opt-in: dist-comm.cpp only calls dist_nd_establish() when
// LLAMA_DIST_TP_RDMA is set to a true value.

struct dist_nd_conn;

// true when this build was compiled with NetworkDirect support
bool dist_nd_available();

// Establishes one bidirectional ND connection with the ring peer.
//
//   tcp_out / tcp_in : the ring sockets already connected to the peer. tcp_out carries the
//                      outbound half of the negotiation, tcp_in the inbound half; tcp_in is
//                      also polled for end-of-stream while blocked on the completion queue.
//                      When one socket serves a single ring edge, pass it as both.
//   listen_side      : true  -> wait for the peer to dial in
//                      false -> dial out to the peer
//   strict_local_addr: refuse to fall back to scanning other ND capable local addresses when
//                      the address behind the ring socket is not itself ND capable.
//
//                      The scan is right for a 2-rank ring, where the one peer is reachable and
//                      any working adapter will do. It is wrong once the ring spans more than
//                      two hosts: an edge running over ordinary Ethernet would scan, find the
//                      unrelated point-to-point adapter, and advertise or dial an address the
//                      peer cannot reach on that edge, turning a clean TCP fallback into a
//                      ND_CONNECT_TIMEOUT_S stall. Under strict mode such an edge fails fast
//                      and stays on TCP, which is the correct answer for a cable that is
//                      not there.
//
// Both ranks agree on the outcome before this returns: the listening side makes the final call
// and tells the other. Returns nullptr with err set when ND is unavailable or negotiation failed.
dist_nd_conn * dist_nd_establish(uintptr_t tcp_out, uintptr_t tcp_in, bool listen_side, std::string & err,
                                 bool strict_local_addr = false);

// Sends a header and its payload as a single ND chunk, so one collective costs one round of
// wire latency rather than two.
bool dist_nd_send2(dist_nd_conn * c, const void * hdr, size_t hdr_size,
                   const void * payload, size_t payload_size, std::string & err);
bool dist_nd_recv(dist_nd_conn * c, void * data, size_t size, std::string & err);
// The 3-rank full mesh's exchange for a payload of any size: sends one message (hdr, then
// payload) to both a and b while receiving one message of the same sizes from each, all four
// directions at once and straight into the caller's buffers, interleaving 256 KiB chunks across
// the two connections. It never blocks on one connection while the other could move. Needs
// LLAMA_DIST_TP_RDMA_MULTICHUNK; a and b must be different connections, and the send buffers
// must not overlap the receive buffers. In full-duplex mode the sends are left to
// dist_nd_send_drain, as after dist_nd_send2.
bool dist_nd_exchange_pair(dist_nd_conn * a, dist_nd_conn * b, const void * hdr, size_t hdr_size,
                           const void * payload, size_t size, void * hdr_from_a, void * from_a,
                           void * hdr_from_b, void * from_b, std::string & err);
// The 4-rank full mesh's large exchange: dist_nd_exchange_pair over n distinct connections at once
// (1 to 4). One message (hdr, then payload) leaves on every connection while one message of the same
// sizes arrives from each, into hdr_from[i] / from[i] for connection i, interleaving 256 KiB chunks
// across all of them. Same rules as the pair: LLAMA_DIST_TP_RDMA_MULTICHUNK, send buffers must not
// overlap receive buffers, and in full-duplex mode the sends are left to dist_nd_send_drain.
bool dist_nd_exchange_n(dist_nd_conn * const * conns, int n, const void * hdr, size_t hdr_size,
                        const void * payload, size_t size, void * const * hdr_from, void * const * from,
                        std::string & err);
// Wait for sends posted but not yet retired. Only needed in full-duplex mode, where dist_nd_send2
// returns as soon as the chunk is posted rather than when it has completed.
bool dist_nd_send_drain(dist_nd_conn * c, std::string & err);

void dist_nd_free(dist_nd_conn * c);
