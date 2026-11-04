#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// NetworkDirect (Windows RDMA) data path for the distributed-TP ring.
//
// This is a transport substitution only. The ring bootstrap, the op_hdr wire framing and the
// send/recv ordering all stay in dist-comm.cpp; only the bytes of the collective payload move
// over NetworkDirect instead of TCP. The ND connection is negotiated out of band over the ring
// sockets that dist-comm.cpp has already established, and every failure path leaves the caller
// on TCP rather than aborting.
//
// dist-comm.cpp calls dist_nd_establish() for an edge when both of its ranks have an ND capable
// address on one subnet (dist_nd_local_addrs) and neither sets LLAMA_DIST_TP_RDMA=0.

struct dist_nd_conn;

// true when this build was compiled with NetworkDirect support
bool dist_nd_available();

// An ND capable local IPv4 address and its on-link prefix length.
struct dist_nd_local_addr {
    uint32_t ipv4;   // network byte order
    uint32_t prefix; // bits
};

// The local IPv4 addresses NetworkDirect can use, with their prefix lengths. Empty when this build
// has no NetworkDirect support or no ND provider is installed.
std::vector<dist_nd_local_addr> dist_nd_local_addrs();

// Sizes the small connection that every edge set up after this call adds next to its 256 KiB
// one: a message of up to `bytes` (the caller's header included) rides it, so a decode exchange
// moves a few pages instead of a whole 256 KiB slot. 0, or a size that does not fit below 256 KiB,
// means no small connection. Call it before the first dist_nd_establish, with the same value on
// every rank: ranks with different values refuse to connect.
void dist_nd_set_small_message(size_t bytes);

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
// Wait for sends posted but not yet retired. dist_nd_send2 returns as soon as its last chunk is
// posted, so call this once the matching receive is done.
bool dist_nd_send_drain(dist_nd_conn * c, std::string & err);

void dist_nd_free(dist_nd_conn * c);
