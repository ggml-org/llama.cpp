
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"

#define GGML_COMMON_IMPL_C
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-hexagon.h"
#include "ggml-impl.h"

#include "htp-utils.h"

#include <domain.h>
#include <remote.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

domain * get_domain(int domain_id) {
    int i    = 0;
    int size = sizeof(supported_domains) / sizeof(domain);

    for (i = 0; i < size; i++) {
        if (supported_domains[i].id == domain_id) {
            return &supported_domains[i];
        }
    }

    return NULL;
}

bool is_valid_domain_id(int domain_id, int compute_only) {
    int i    = 0;
    int size = sizeof(supported_domains) / sizeof(domain);

    if (compute_only) {
        return is_CDSP(domain_id);
    }

    for (i = 0; i < size; i++) {
        if (supported_domains[i].id == domain_id) {
            return true;
        }
    }

    return false;
}

int get_domains_info(char * domain_type, int * num_domains, fastrpc_domain ** domains_info) {
    int nErr    = AEE_SUCCESS;
    int ss_info = 0;
    if (domain_type != NULL) {
        if (strcmp(domain_type, "LPASS") == 0) {
            ss_info = FASTRPC_LPASS;
        } else if (strcmp(domain_type, "HPASS") == 0) {
            ss_info = FASTRPC_HPASS;
        } else {
            ss_info = FASTRPC_NSP;
        }
    }
    system_req_payload req  = { 0 };
    req.id                  = FASTRPC_GET_DOMAINS;
    req.sys.domains         = NULL;
    fastrpc_domain * domain = NULL;
    if (ss_info != 0) {
        req.sys.flags = DOMAINS_LIST_FLAGS_SET_TYPE(req.sys.flags, ss_info);
    } else {
        req.sys.flags = 0;
    }
#ifdef _WIN32
    nErr = AEE_EUNSUPPORTED;
    goto bail;
#endif
    if (remote_system_request) {
        nErr = remote_system_request(&req);
        if (nErr != AEE_SUCCESS) {
            GGML_LOG_ERROR("Failure in remote_system_request call: %d.\n", nErr);
            goto bail;
        }
        // Allocate memory for domain-info array
        req.sys.max_domains = req.sys.num_domains;
        if ((req.sys.domains = calloc(req.sys.num_domains, sizeof(fastrpc_domain))) == NULL) {
            nErr = AEE_ENOMEMORY;
            GGML_LOG_ERROR("Unable to allocate memory for req.sys.domains");
            goto bail;
        }

        nErr = remote_system_request(&req);
        if (nErr != AEE_SUCCESS) {
            GGML_LOG_ERROR("Failure in remote_system_request call: %d.\n", nErr);
            goto bail;
        }

        for (int i = 0; i < req.sys.num_domains; i++) {
            // Verify that only requested type domains were returned
            domain = &req.sys.domains[i];
            if (domain->type != ss_info && domain_type != NULL) {
                nErr = -1;
                GGML_LOG_ERROR("Incorrect data received from remote_system_request.\n");
                goto bail;
            }
        }
        *domains_info = req.sys.domains;
        *num_domains  = req.sys.num_domains;
    } else {
        nErr = AEE_EUNSUPPORTED;
        goto bail;
    }
bail:
    if (nErr && !req.sys.domains) {
        free(req.sys.domains);
    }
    return nErr;
}

int get_effective_domain_id(char * domain_name, int session_id, int * effec_domain_id) {
    int                              err  = 0;
    remote_rpc_effective_domain_id_t sess = { 0 };

    sess.domain_name     = domain_name;
    sess.domain_name_len = strlen(domain_name);
    sess.session_id      = session_id;

    err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, &sess, sizeof(sess));
    if (err) {
        GGML_LOG_ERROR("Error 0x%x: failed to get effective domain id for %s, session id %d\n", err, sess.domain_name,
               session_id);
        return err;
    }

    *effec_domain_id = sess.effective_domain_id;
    return err;
}

int get_dsp_support(int * domain) {
    int nErr = AEE_SUCCESS;
    *domain  = CDSP_DOMAIN_ID;  // DSP domain default value is CDSP_DOMAIN_ID

    if (remote_handle_control) {
        struct remote_dsp_capability dsp_capability_domain = { CDSP_DOMAIN_ID, DOMAIN_SUPPORT, 0 };
        nErr = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
        if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGML_LOG_ERROR("\nFastRPC Capability API is not supported on this device\n");
            goto bail;
        }

        if (dsp_capability_domain.capability == 0) {
            dsp_capability_domain.domain       = ADSP_DOMAIN_ID;  // Check for ADSP support.
            dsp_capability_domain.attribute_ID = DOMAIN_SUPPORT;
            dsp_capability_domain.capability   = 0;
            nErr                               = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain,
                                                                       sizeof(struct remote_dsp_capability));
            if (dsp_capability_domain.capability) {
                *domain = ADSP_DOMAIN_ID;  // For targets like Agatti (not having cDSP), domain is ADSP_DOMAIN_ID
            }
        }

        if (nErr != AEE_SUCCESS) {
            GGML_LOG_ERROR("\nget_dsp_support failed with Error 0x%x\n", nErr);
            goto bail;
        }
    } else {
        nErr = AEE_EUNSUPPORTEDAPI;
        GGML_LOG_ERROR("remote_dsp_capability interface is not supported on this device\n");
    }

bail:
    return nErr;
}

int get_vtcm_info(int domain, uint32_t * capability, uint32_t attr) {
    int nErr    = AEE_SUCCESS;
    *capability = 0;

    if (attr == VTCM_PAGE || attr == VTCM_COUNT) {
    } else {
        nErr = AEE_EBADPARM;
        GGML_LOG_ERROR("Unsupported attr. Only VTCM_PAGE and VTCM_COUNT supported\n");
        goto bail;
    }
    if (remote_handle_control) {
        if (domain == ADSP_DOMAIN_ID || domain == CDSP_DOMAIN_ID) {
            /*
            * Query the DSP for VTCM information
            * Since the ADSP does not have a dedicated VTCM, we expect the output to be 0
            */
            struct remote_dsp_capability dsp_capability_vtcm_dsp;
            dsp_capability_vtcm_dsp.domain       = (uint32_t) domain;
            dsp_capability_vtcm_dsp.attribute_ID = attr;
            dsp_capability_vtcm_dsp.capability   = (uint32_t) 0;
            nErr                                 = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_vtcm_dsp,
                                                                         sizeof(struct remote_dsp_capability));
            if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGML_LOG_ERROR("\nFastRPC Capability API is not supported on this device\n");
                GGML_LOG_ERROR("Running the usecase without checking the capability\n");
                nErr = AEE_SUCCESS;
                goto bail;
            } else if (nErr == AEE_SUCCESS) {
                *capability = dsp_capability_vtcm_dsp.capability;
            } else {
                GGML_LOG_ERROR("\nget_vtcm_info failed with Error 0x%x\n", nErr);
                goto bail;
            }
        } else {
            nErr = AEE_EUNSUPPORTED;
            GGML_LOG_ERROR("Unsupported domain %d\n", domain);
            goto bail;
        }
    } else {
        nErr = AEE_EUNSUPPORTEDAPI;
        GGML_LOG_ERROR("remote_dsp_capability interface is not supported on this device\n");
    }

bail:
    return nErr;
}

bool is_unsignedpd_supported(int domain_id) {
    int nErr = AEE_SUCCESS;
    if (remote_handle_control) {
        struct remote_dsp_capability dsp_capability_domain = { domain_id, UNSIGNED_PD_SUPPORT, 0 };
        nErr = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
        if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGML_LOG_ERROR("\nFastRPC Capability API is not supported on this device. Falling back to signed pd.\n");
            return false;
        }
        if (nErr) {
            GGML_LOG_ERROR("\nERROR 0x%x: FastRPC Capability API failed. Falling back to signed pd.", nErr);
            return false;
        }
        if (dsp_capability_domain.capability == 1) {
            return true;
        }
    } else {
        nErr = AEE_EUNSUPPORTEDAPI;
        GGML_LOG_ERROR("remote_dsp_capability interface is not supported on this device. Falling back to signed pd.\n");
        return false;
    }
    return false;
}

bool get_unsignedpd_support(void) {
    return is_unsignedpd_supported(CDSP_DOMAIN_ID);
}

bool is_async_fastrpc_supported(int domain) {
    int nErr = AEE_SUCCESS;
    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
            /*
            * Query the DSP for ASYNC_FASTRPC_SUPPORT information
            * Async fastrpc is supported only on CDSP
            */
            struct remote_dsp_capability dsp_capability_async_support;
            dsp_capability_async_support.domain       = (uint32_t) domain;
            dsp_capability_async_support.attribute_ID = ASYNC_FASTRPC_SUPPORT;
            dsp_capability_async_support.capability   = (uint32_t) 0;
            nErr = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_async_support,
                                         sizeof(struct remote_dsp_capability));
            if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGML_LOG_ERROR("\nFastRPC Capability API is not supported on this device\n");
                GGML_LOG_ERROR("Running the usecase without checking the capability\n");
                nErr = AEE_SUCCESS;
                goto bail;
            } else if (dsp_capability_async_support.capability == 1) {
                return true;
            }
            if (nErr != AEE_SUCCESS) {
                GGML_LOG_ERROR("\nis_async_fastrpc_supported failed with Error 0x%x\n", nErr);
                goto bail;
            }
        } else {
            nErr = AEE_EUNSUPPORTED;
            GGML_LOG_ERROR("Async fastrpc is not supported on domain %d\n", domain);
            goto bail;
        }
    } else {
        nErr = AEE_EUNSUPPORTEDAPI;
        GGML_LOG_ERROR("remote_dsp_capability interface is not supported on this device\n");
    }

bail:
    return false;
}

bool is_status_notification_supported(int domain) {
    int nErr = AEE_SUCCESS;

    if (remote_handle_control) {
        /*
        * Query the DSP for STATUS_NOTIFICATION_SUPPORT information
        * DSP User PD status notification Support
        */
        struct remote_dsp_capability dsp_capability_status_notification_support;
        dsp_capability_status_notification_support.domain       = (uint32_t) domain;
        dsp_capability_status_notification_support.attribute_ID = STATUS_NOTIFICATION_SUPPORT;
        dsp_capability_status_notification_support.capability   = (uint32_t) 0;
        nErr = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_status_notification_support,
                                     sizeof(struct remote_dsp_capability));
        if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGML_LOG_ERROR("\nFastRPC Capability API is not supported on this device\n");
            GGML_LOG_ERROR("Running the usecase without checking the capability\n");
            nErr = AEE_SUCCESS;
            goto bail;
        } else if (dsp_capability_status_notification_support.capability == 1) {
            return true;
        }
        if (nErr != AEE_SUCCESS) {
            GGML_LOG_ERROR("\nis_status_notification_supported failed with Error 0x%x\n", nErr);
            goto bail;
        }
    } else {
        nErr = AEE_EUNSUPPORTEDAPI;
        GGML_LOG_ERROR("remote_dsp_capability interface is not supported on this device\n");
    }

bail:
    return false;
}

int get_hmx_support_info(int domain, uint32_t * capability, uint32_t attr) {
    int nErr    = AEE_SUCCESS;
    *capability = 0;

    if (attr != HMX_SUPPORT_SPATIAL && attr != HMX_SUPPORT_DEPTH) {
        nErr = AEE_EBADPARM;
        GGML_LOG_ERROR("Unsupported attr. Only HMX_SUPPORT_SPATIAL and HMX_SUPPORT_DEPTH supported\n");
        goto bail;
    }
    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
            /*
            * Query the DSP for HMX SUPPORT information
            * HMX is supported on CDSP only
            */
            struct remote_dsp_capability dsp_capability_hmx_dsp;
            dsp_capability_hmx_dsp.domain       = (uint32_t) domain;
            dsp_capability_hmx_dsp.attribute_ID = attr;
            dsp_capability_hmx_dsp.capability   = (uint32_t) 0;
            nErr                                = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hmx_dsp,
                                                                        sizeof(struct remote_dsp_capability));
            if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGML_LOG_ERROR("\nFastRPC Capability API is not supported on this device\n");
                GGML_LOG_ERROR("Running the usecase without checking the capability\n");
                nErr = AEE_SUCCESS;
                goto bail;
            } else if (nErr == AEE_SUCCESS) {
                *capability = dsp_capability_hmx_dsp.capability;
            } else {
                GGML_LOG_ERROR("\nget_hmx_support_info failed with Error 0x%x\n", nErr);
                goto bail;
            }
        } else {
            nErr = AEE_EUNSUPPORTED;
            GGML_LOG_ERROR("HMX support is not there for domain %d\n", domain);
            goto bail;
        }
    } else {
        nErr = AEE_EUNSUPPORTEDAPI;
        GGML_LOG_ERROR("remote_dsp_capability interface is not supported on this device\n");
    }

bail:
    return nErr;
}

int get_hex_arch_ver(int domain, int * arch) {
    if (!remote_handle_control) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        return AEE_EUNSUPPORTEDAPI;
    }

    struct remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    int err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, sizeof(arch_ver));
    if ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        return AEE_EUNSUPPORTEDAPI;
    }

    if (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        return err;
    }

    switch (arch_ver.capability & 0xff) {
        case 0x68:
            *arch = 68;
            return 0;
        case 0x69:
            *arch = 69;
            return 0;
        case 0x73:
            *arch = 73;
            return 0;
        case 0x75:
            *arch = 75;
            return 0;
        case 0x79:
            *arch = 79;
            return 0;
        case 0x81:
            *arch = 81;
            return 0;
    }
    return -1;
}

int get_hvx_support_info(int domain, uint32_t * capability, uint32_t attr) {
    int nErr    = AEE_SUCCESS;
    *capability = 0;

    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
            /*
            * Query the DSP for HVX SUPPORT information
            * HVX is supported on CDSP only
            */
            struct remote_dsp_capability dsp_capability_hvx_dsp;
            dsp_capability_hvx_dsp.domain       = (uint32_t) domain;
            dsp_capability_hvx_dsp.attribute_ID = attr;
            dsp_capability_hvx_dsp.capability   = (uint32_t) 0;
            nErr                                = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hvx_dsp,
                                                                        sizeof(struct remote_dsp_capability));
            if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGML_LOG_ERROR("\nFastRPC Capability API is not supported on this device\n");
                GGML_LOG_ERROR("Running the usecase without checking the capability\n");
                nErr = AEE_SUCCESS;
                goto bail;
            } else if (nErr == AEE_SUCCESS) {
                *capability = dsp_capability_hvx_dsp.capability;
            } else {
                GGML_LOG_ERROR("\nget_hvx_support_info failed with Error 0x%x\n", nErr);
                goto bail;
            }
        } else {
            nErr = AEE_EUNSUPPORTED;
            GGML_LOG_ERROR("HVX support is not available on domain %d\n", domain);
            goto bail;
        }
    } else {
        nErr = AEE_EUNSUPPORTEDAPI;
        GGML_LOG_ERROR("remote_dsp_capability interface is not supported on this device\n");
    }

bail:
    return nErr;
}
