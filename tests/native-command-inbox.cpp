#define STRATA_COMMAND_INBOX_TEST_HOOKS
#include "strata/common/command_inbox.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using Inbox = strata::common::CommandInbox;
using Clock = std::chrono::steady_clock;

void say(const char* text) { std::printf("%s\n", text); std::fflush(stdout); }

void command(Inbox& inbox, const std::string& text) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ull; }
    const bool cancelled = inbox.cancelled().load();
    std::printf("CMD %d %zu %016llx\n", cancelled, text.size(), (unsigned long long) hash);
    std::fflush(stdout);
}

void wait_closed(const Inbox& inbox) {
    const auto until = Clock::now() + std::chrono::seconds(5);
    while (!inbox.input_closed()) {
        if (Clock::now() > until) throw std::runtime_error("reader did not reach EOF/QUIT");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int run(const std::string& mode) {
    if (mode == "init-error") {
        say("READY");
        Inbox::fail_next_initialization_for_test();
        Inbox inbox;
        // Windows propagates constructor startup failures immediately; POSIX
        // reports a read failure through next(). Both must join on unwind.
        std::string line;
        (void) inbox.next(line);
        throw std::runtime_error("injected initialization unexpectedly succeeded");
    }
    if (mode == "cycles") {
        say("READY");
#if defined(_WIN32)
        DWORD initial_handles = 0;
#endif
        for (int i = 0; i < 40; ++i) {
            {
                Inbox inbox;
                std::string line;
                if (!inbox.next(line) || line != "CYCLE") throw std::runtime_error("cycle command missing");
            }
#if defined(_WIN32)
            DWORD handles = 0;
            if (!GetProcessHandleCount(GetCurrentProcess(), &handles)) throw std::runtime_error("handle count unavailable");
            if (i == 0) initial_handles = handles;
            if (handles > initial_handles + 2) throw std::runtime_error("reader handles leaked across lifetimes");
#endif
            std::printf("CYCLE %d\n", i); std::fflush(stdout);
        }
        return 0;
    }
    Inbox inbox;
    say("READY");
    if (mode == "queue") wait_closed(inbox);
    std::string line;
    while (inbox.next(line)) {
        if (mode == "error") {
            if (line != "ERROR") throw std::runtime_error("expected ERROR");
            say("LEAVING");
            throw std::runtime_error("simulated model error");
        }
        if (mode == "return") {
            if (line != "RETURN") throw std::runtime_error("expected RETURN");
            say("LEAVING");
            return 0;
        }
        if (mode == "stop") {
            if (line != "STOP_READER") throw std::runtime_error("expected STOP_READER");
            inbox.stop();
            inbox.stop(); // idempotent; destructor calls it once more
            if (inbox.next(line)) throw std::runtime_error("stopped inbox returned a command");
            say("STOPPED");
            return 0;
        }
        command(inbox, line);
        if (mode == "cancel" && line == "GEN first") {
            say("START");
            const auto until = Clock::now() + std::chrono::seconds(5);
            while (!inbox.cancelled().load()) {
                if (Clock::now() > until) throw std::runtime_error("STOP did not cancel running work");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            say("CANCELLED");
        }
        if (line == "QUIT") break;
    }
    say("END");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return run(argc > 1 ? argv[1] : "echo"); }
    catch (const std::exception& error) {
        std::printf("ERROR %s\n", error.what()); std::fflush(stdout);
        return 13;
    }
}
