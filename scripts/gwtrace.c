/* LD_PRELOAD tracer for the aclshmem GW exchange (r9 forensics).
 * Logs entry/exit of the store/engine/transport functions around the
 * _8_GW descriptor exchange: keys, payload sizes, indices, return codes.
 * Output goes to stderr with a [gwtrace] prefix; single write() per line.
 */
#define _GNU_SOURCE
#include <string.h>
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


#include <signal.h>
#include <ucontext.h>
static void segv_handler(int sig, siginfo_t* si, void* uc) {
    ucontext_t* c = uc;
    char b[256];
    int n = snprintf(b, sizeof(b),
        "[gwtrace] !! SIGSEGV addr=%p pc=%p (offset in lib?) lr=%p\n",
        si->si_addr, (void*)c->uc_mcontext.pc, (void*)c->uc_mcontext.regs[30]);
    if (n > 0) { ssize_t w = write(2, b, n); (void)w; }
    { /* locate pc & lr in /proc/self/maps */
        FILE* f = fopen("/proc/self/maps", "r");
        if (f) {
            char line[512];
            unsigned long lo, hi;
            while (fgets(line, sizeof(line), f)) {
                if (sscanf(line, "%lx-%lx", &lo, &hi) == 2) {
                    unsigned long pcs[2] = {c->uc_mcontext.pc, c->uc_mcontext.regs[30]};
                    for (int k = 0; k < 2; k++)
                        if (pcs[k] >= lo && pcs[k] < hi) {
                            char b2[600];
                            int n2 = snprintf(b2, sizeof(b2), "[gwtrace]   pc%d maps: %s", k, line);
                            if (n2 > 0) { ssize_t w2 = write(2, b2, n2 > 0 ? (size_t)n2 : 0); (void)w2; }
                        }
                }
            }
            fclose(f);
        }
    }
    _exit(88);
}
__attribute__((constructor)) static void segv_install(void) {
    struct sigaction sa = {0};
    sa.sa_sigaction = segv_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, 0);
    sigaction(SIGBUS, &sa, 0);
}


static void hexdump(const char* tag, const unsigned char* p, size_t n) {
    size_t lim = n > 256 ? 256 : n;
    tlog("[gwtrace] %s len=%zu", tag, n);
    for (size_t i = 0; i < lim; i += 16) {
        char b[128]; size_t o = 0;
        o += (size_t)snprintf(b + o, sizeof(b) - o, "[gwtrace]  %04zx ", i);
        for (size_t j = 0; j < 16 && i + j < lim; j++)
            o += (size_t)snprintf(b + o, sizeof(b) - o, "%02x", p[i + j]);
        o += (size_t)snprintf(b + o, sizeof(b) - o, "  ");
        for (size_t j = 0; j < 16 && i + j < lim; j++) {
            unsigned char c = p[i + j];
            b[o++] = (c >= 32 && c < 127) ? c : '.';
        }
        b[o] = 0;
        tlog("%s", b);
    }
    if (n > lim) tlog("[gwtrace]  ... (%zu more)", n - lim);
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
        if (!T##_real) {                                                         \
        T##_real = (T##_fn)dlsym(RTLD_NEXT, #T);                             \
        if (!T##_real) {                                                     \
            static const char* libs[] = {"aclshmem_bootstrap_config_store.so", "libshmem.so", NULL}; \
            for (int i = 0; libs[i] && !T##_real; i++) {                     \
                void* h = dlopen(libs[i], RTLD_LAZY | RTLD_NOLOAD);          \
                if (!h) h = dlopen(libs[i], RTLD_LAZY | RTLD_GLOBAL);        \
                if (h) T##_real = (T##_fn)dlsym(h, #T);                      \
            }                                                                \
        }                                                                    \
        if (!T##_real) { tlog("[gwtrace] !! cannot resolve " #T); *(volatile int*)0 = 0; } \
    }                                                                        \
        { log_entry }                                                          \
        long r = T##_real(a0, a1, a2, a3, a4, a5, a6);                        \
        { log_exit }                                                           \
        return r;                                                              \
    }

/* --- store engine / tcp config store --- */
FWD7(ga_wrap, _ZN3shm5store18SmemNetGroupEngine14GroupAllGatherEPKcjPcj,
     tlog("[gwtrace] GroupAllGather klen=%u out=%p outlen=%u", (unsigned)(uintptr_t)a2, a3, (unsigned)(uintptr_t)a4);
     if ((unsigned)(uintptr_t)a2 > 50) hexdump("GA name buf", (const unsigned char*)a1, (unsigned)(uintptr_t)a2);,
     tlog("[gwtrace] GroupAllGather ret=%ld", r);)

FWD7(append_wrap, _ZN3shm5store14TcpConfigStore6AppendERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIhSaIhEERm,
     { size_t n; const char* k = cstr(a1, &n); tlog("[gwtrace] Append key=%s dlen=%zu", k, vsize(a2));
       if (strstr(k, "_8_") || strstr(k, "_4_GA")) { const unsigned char* const* v = a2; hexdump("Append payload", v[0], vsize(a2)); } },
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

FWD7(set_wrap, _ZN3shm5store14TcpConfigStore3SetERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIhSaIhEE,
     tlog("[gwtrace] Set key=%s dlen=%zu", cstr(a1, 0), vsize(a2));,
     tlog("[gwtrace] Set ret=%ld", r);)

FWD7(plugin_wrap, aclshmemi_bootstrap_plugin_init,
     tlog("[gwtrace] ==== bootstrap_plugin_init enter");,
     tlog("[gwtrace] ==== bootstrap_plugin_init ret=%ld", r);)
