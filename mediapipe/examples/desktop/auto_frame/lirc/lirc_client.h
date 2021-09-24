#include <string>

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <sysexits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <syslog.h>
#include <pwd.h>
#include <ctype.h>
#include <pthread.h>
#include <vector>

using std::string;

class LircClient {

private:
	
	pthread_t lirc_thread;
	bool    m_isRunning;
    void (*m_callback)(std::vector<string>) ;
	int     m_sockfd = -1;
	string  m_device_path;
	
public:

	LircClient() { }
    ~LircClient();
	bool Open(string device_addr, void (*callback)(std::vector<string>));
	bool Close(void);
	void main_loop(void);
	
};