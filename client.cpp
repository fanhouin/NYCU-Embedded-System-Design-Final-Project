#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <errno.h>


struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );

typedef struct
{
    int *count;
    cv::Mat frame;
} Args;

int start = 0;
char encodeImg[1000000];
char decodeImg[1000000];
int bytes = 0;
int sockfd;
float time_use = 0;
struct timeval start_t;
struct timeval end_t;


struct sockaddr_in	cli_addr, serv_addr;

int recvlen(int fd, char *ptr, int maxlen){
    int rc;
    int total_read = 0;
    char *buf = ptr;
    char c;

    while(1){
        rc = recv(fd, &c, 1, 0);
        if(rc == -1){
            if(errno == EINTR) continue;
            else return -1;
        }
        else if(rc == 0){
            if(total_read == 0) return 0;
            else break;
        }
        else{
            if(total_read <= maxlen){
                total_read++;
				*buf++ = c;
				if(total_read == maxlen) break;
            }
        }
    }

    *buf = '\0';
    return total_read;
}

int recvimg(int fd, char *ptr, int maxlen){
    int rc;
    int total_read = 0;
    char *buf = ptr;
    char c;

    while(1){
        rc = recv(fd, &c, 1, 0);
        if(rc == -1){
            if(errno == EINTR) continue;
            else return -1;
        }
        else{
            if(total_read <= maxlen){
				total_read++;
				*buf++ = c;
                if(total_read == maxlen) break;
            }
        }
    }
    *buf = '\0';
    return total_read;
}


void* getkey(void *arg) {
	char c;
	while(1){
		system("stty raw");
		c = getchar();
		system("stty cooked");
		if(c == 'o'){
		 	start = 1;
		}
	}
}



int main ( int argc, const char *argv[] )
{   
	cv::VideoCapture camera(2);
	cv::Mat frame;
	cv::Size2f frame_size;
 
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("192.168.1.10");
    serv_addr.sin_port = htons(6666);
	
	if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0){
		perror("[-]Error in socket");
        exit(1);
	}
	printf("[+]Server socket created. \n");

	if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0){
		perror("[-]Error in Connecting");
        exit(1);
	}
	printf("[+]Connected to server.\n");

 
    int count = 0;
    Args arg;

    // check if video stream device is opened success or not
    if(!camera.isOpened())
    {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }
	printf("open canera\n");

	
 	
	pthread_t t;
	pthread_create(&t, NULL, getkey, (void *)NULL);


    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");
	cv::Mat proc_img;
	cv::Size2f proc_img_size;
	char lenbuf[20];
	char str[30];

    while ( true )
    {
		if(start) break;
		bool ret = camera.read(proc_img);
		cv::cvtColor(proc_img, proc_img, cv::COLOR_BGR2BGR565);		
		proc_img_size = proc_img.size();
		
		for ( int y = 0; y < proc_img_size.height; y++ ){
			ofs.seekp(y * fb_info.xres_virtual * fb_info.bits_per_pixel / 8);
			const char * data = proc_img.ptr<char>(y);
		    int size = proc_img_size.width * fb_info.bits_per_pixel / 8;
		    ofs.write(data, size);
		}

	}

    while ( true )
    {
		gettimeofday(&start_t, NULL);
		bool ret = camera.read(frame);
		std::vector<unsigned char> data_encode;
	    std::vector<int> quality;
		cv::resize(frame, frame, cv::Size(320, 240), 0 ,0, cv::INTER_LINEAR);
	    cv::imencode(".png", frame, data_encode);

 		int nSize = data_encode.size();
	    for (int i = 0; i < nSize; i++){
	        encodeImg[i] = data_encode[i];
	    }
		memset(lenbuf, '\0', sizeof(lenbuf));
		sprintf(lenbuf, "%d", nSize);

		//send image length
		if ((bytes = send(sockfd, lenbuf, 10, 0)) < 0){
		    std::cerr << "bytes = " << bytes << std::endl;
		    break;
		}
		//send processed image
		if ((bytes = send(sockfd, encodeImg, nSize, 0)) < 0){
		    std::cerr << "bytes = " << bytes << std::endl;
		    break;
		}

		std::vector<unsigned char> decode;
		memset(lenbuf, '\0', sizeof(lenbuf));
		int n = recv(sockfd, lenbuf, 10, 0);
		if(n == -1) continue;
		int img_len = atoi(lenbuf);
		if(img_len == 0) continue;

		int nn = recvimg(sockfd, decodeImg, img_len);
		//int nn = recv(sockfd, decodeImg, img_len, 0);
		//std::cout << nn << std::endl;
		if(nn == -1) continue;

    	int pos = 0;
        while (pos < nn){
            decode.push_back(decodeImg[pos++]);
        }
		proc_img = cv::imdecode(decode, CV_LOAD_IMAGE_COLOR);
		cv::resize(proc_img, proc_img, cv::Size(640, 480), 0 ,0, cv::INTER_LINEAR);
		
		gettimeofday(&end_t, NULL);
		time_use = (end_t.tv_sec - start_t.tv_sec) * 1000 + (end_t.tv_usec - start_t.tv_usec) / 1000;
		memset(str, '\0', sizeof(str));
		
		sprintf(str, "%.3fms", time_use);
	
		cv::putText(proc_img, //target image
            str, //text
            cv::Point(10, 30), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(118, 185, 0), //font color
            2);
		cv::cvtColor(proc_img, proc_img, cv::COLOR_BGR2BGR565);		
		proc_img_size = proc_img.size();
		
		for ( int y = 0; y < proc_img_size.height; y++ ){
			ofs.seekp(y * fb_info.xres_virtual * fb_info.bits_per_pixel / 8);
			const char * data = proc_img.ptr<char>(y);
		    int size = proc_img_size.width * fb_info.bits_per_pixel / 8;
		    ofs.write(data, size);
		}
	}
    camera.release();
    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open device with linux system call "open()"
    // https://man7.org/linux/man-pages/man2/open.2.html
    int fd = open(framebuffer_device_path,O_RDWR);
    if (fd == -1) {
        perror("open file");
        exit(EXIT_FAILURE);
    }

    // get attributes of the framebuffer device thorugh linux system call "ioctl()".
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt
    ioctl(fd,FBIOGET_VSCREENINFO,&screen_info);


    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    fb_info.xres_virtual = screen_info.xres_virtual;     // 8
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;    // 16

    return fb_info;
};
