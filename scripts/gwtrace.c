/* LD_PRELOAD tracer for the aclshmem GW exchange (r9 forensics).
 * Logs entry/exit of the store/engine/transport functions around the
 * _8_GW descriptor exchange: keys, payload sizes, indices, return codes.
 * Output goes to stderr with a [gwtrace] prefix; single write() per line.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

static void tlog(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf) - 1, fmt, ap);
    va_end(ap);
    if (n > 0) {
        if (n > (int)sizeof(buf) - 2) n = sizeof(buf) - 2;
        buf[n] = '\n';
        write(2, buf, n + 1);
    }
}

/* libstdc++ cxx11 string: {ptr, size, sso[16]} */
static const char* cstr(const void* sp, size_t* len) {
    const unsigned char* p = sp;
    size_t n = *(const size_t*)(p + 8);
    if (len) *len = n;
    return n > 15 ? *(const char* const*)p : (const char*)(p + 16);
}

/* std::vector: {begin,end,cap} */
static size_t vsize(const void* vp) {
    const unsigned char* const* v = vp;
    return (size_t)(v[1] - v[0]);
}

#define FWD7(name, T, log_entry, log_exit)                                   \
    typedef long (*T##_fn)(void*, void*, void*, void*, void*, void*, void*); \
    static T##_fn T##_real;                                                   \
    long name(void* a0, void* a1, void* a2, void* a3, void* a4, void* a5, void* a6) __asm__(#T); \
    long name(void* a0, void* a1, void* a2, void* a3, void* a4, void* a5, void* a6) { \
        if (!T##_real) T##_real = (T##_fn)dlsym(RTLD_NEXT, #T);               \
        { log_entry }                                                          \
        long r = T##_real(a0, a1, a2, a3, a4, a5, a6);                        \
        { log_exit }                                                           \
        return r;                                                              \
    }

/* --- store engine / tcp config store --- */
FWD7(ga_wrap, _ZN3shm5store18SmemNetGroupEngine14GroupAllGatherEPKcjPcj,
     tlog("[gwtrace] GroupAllGather key=%s", (char*)a1); tlog("[gwtrace]   in klen=%u out=%p outlen=%u", (unsigned)(uintptr_t)a2, a3, (unsigned)(uintptr_t)a4);,
     tlog("[gwtrace] GroupAllGather ret=%ld", r);)

FWD7(append_wrap, _ZN3shm5store14TcpConfigStore6AppendERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIhSaIhEERm,
     { size_t n; tlog("[gwtrace] Append key=%s dlen=%zu", cstr(a1, &n), vsize(a2)); },
     tlog("[gwtrace] Append ret=%ld", r);)

FWD7(add_wrap, _ZN3shm5store14TcpConfigStore3AddERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEElRl,
     tlog("[gwtrace] Add key=%s v=%ld", cstr(a1, 0), (long)a2);,
     tlog("[gwtrace] Add ret=%ld (*out=%ld)", r, *(long*)a3);)

FWD7(getreal_wrap, _ZN3shm5store14TcpConfigStore7GetRealERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIhSaIhEEl,
     tlog("[gwtrace] GetReal key=%s timeout=%ld", cstr(a1, 0), (long)a3);,
     tlog("[gwtrace] GetReal ret=%ld (out dlen=%zu)", r, vsize(a2));)

FWD7(get_wrap, _ZN3shm5store11ConfigStore3GetERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS7_l,
     tlog("[gwtrace] ConfigStore::Get key=%s timeout=%ld", cstr(a1, 0), (long)a3);,
     tlog("[gwtrace] ConfigStore::Get ret=%ld", r);)

/* --- transport layer (libshmem.so) --- */
FWD7(exch_wrap, _ZNK3shm9transport6device20UdmaTransportManager27ExchangeEndpointDescriptorsERNS2_16EndpointExchangeE,
     tlog("[gwtrace] >> ExchangeEndpointDescriptors enter (this=%p)", a0);,
     tlog("[gwtrace] << ExchangeEndpointDescriptors ret=%ld", r);)

FWD7(gather_wrap, _ZN3shm9transport6device20UdmaTransportManager20GatherEndpointChunksERKSt6vectorINS2_21ExchangedEndpointDescESaIS4_EEjjRNS2_16EndpointExchangeE,
     tlog("[gwtrace] GatherEndpointChunks desc.size=%zu u1=%u u2=%u", vsize(a1), (unsigned)(uintptr_t)a2, (unsigned)(uintptr_t)a3);,
     tlog("[gwtrace] GatherEndpointChunks ret=%ld", r);)

FWD7(plugin_wrap, aclshmemi_bootstrap_plugin_init,
     tlog("[gwtrace] ==== bootstrap_plugin_init enter");,
     tlog("[gwtrace] ==== bootstrap_plugin_init ret=%ld", r);)
