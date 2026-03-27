#pragma once

// Lloyd-Max codebooks for the TurboQuant CPU path.

static const float turboq_codebook_2bit[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

static const float turboq_codebook_3bit[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f,
};

static const float turboq_codebook_4bit[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

static const float turboq_boundaries_2bit[3] = {
    -0.9816f, 0.0000f, 0.9816f,
};

static const float turboq_boundaries_3bit[7] = {
    -1.7480f, -1.0500f, -0.5006f, 0.0000f,
     0.5006f,  1.0500f,  1.7480f,
};

static const float turboq_boundaries_4bit[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5225f, -0.2583f,  0.0000f,
     0.2583f,  0.5225f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f,
};
