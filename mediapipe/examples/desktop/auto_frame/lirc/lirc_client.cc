#include "lirc_client.h"

#include <pthread.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static pthread_mutex_t lirc_sync = PTHREAD_MUTEX_INITIALIZER;

extern "C" void *extf(void *This) {
    static_cast<LircClient *>(This)->main_loop();
}

LircClient::~LircClient() { LircClient::Close(); }

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

bool LircClient::Open(std::string device_addr,
                      void (*callback)(std::vector<std::string>)) {
    if (callback == NULL) {
        std::cout << "ERROR : need to provide callback function" << std::endl;
        return false;
    }
    m_device_path = device_addr;
    m_callback = callback;
    const char *lircpath = m_device_path.c_str();

    struct sockaddr_un sa = {0};
    m_sockfd = socket(AF_UNIX, SOCK_STREAM, 0);

    if (m_sockfd < 0) {
        fprintf(stderr, "Unable to create an AF_UNIX socket: %s\n",
                strerror(errno));
        return false;
    }

    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, lircpath, sizeof sa.sun_path - 1);

    if (connect(m_sockfd, (struct sockaddr *)&sa, sizeof(sa)) == -1) {
        std::cout << "Unable to connect to socket: " << device_addr
                  << std::endl;
        return -errno;
    }

    m_isRunning = true;

    if (pthread_create(&lirc_thread, NULL, &extf, this)) {
        fprintf(stderr, "Can't create LircClient thread");
        return false;
    }

    // pthread_join(lirc_thread, NULL);
    return true;
}

bool LircClient::Close() {
    m_isRunning = false;

    if (m_sockfd >= 0) {
        std::cout << "LircClient::Close() m_sockfd" << std::endl;
        shutdown(m_sockfd, SHUT_RDWR);
        close(m_sockfd);
    }

    pthread_mutex_destroy(&lirc_sync);
    pthread_join(lirc_thread, NULL);

    return true;
}

void LircClient::main_loop(void) {
    if (m_sockfd < 0) return;
    char buffer[1024] = {0};
    int read_count = 0;

    while (m_isRunning) {
        read_count = read(m_sockfd, buffer, 1024);
        if (read_count <= 0) break;

        buffer[read_count + 1] = '\0';
        std::vector<std::string> recvd = split(buffer, ' ');
        m_callback(recvd);
    }
    std::cout << "Closing socket" << std::endl;
    close(m_sockfd);
}