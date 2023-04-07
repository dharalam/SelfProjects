#include <unistd.h>
#include <signal.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <wait.h>
#include <string.h>
#include <pwd.h>
#include <limits.h>
#include "header.h"
// Author: Dimitrios Haralampopoulos
// I pledge my honor that I have abided by the Stevens Honor System

#define MAXARGS (256)
#define MAXLINE (256)

void chandler(int sig){
	pid_t pid;
	while ((pid = waitpid(-1, NULL, WNOHANG)) > 0)
		printf("\npid %d done\n", pid);	
}
void shandler(int sig){
	printf("\nError: Cannot exit shell with command ^C, try exit\n");
    return;	
}

void shinstaller(){
	if (signal(SIGINT, shandler) == SIG_ERR)
		fprintf(stderr, "Error: Cannot register signal handler. %s.\n", strerror(errno)); 
	if (signal(SIGCHLD, chandler) == SIG_ERR)
		fprintf(stderr, "Error: Cannot register signal handler. %s.\n", strerror(errno)); 
}


int parseline (char *buf, char **argv){
    int i = 0;
    int bg = 0;
    if (strcmp(buf, "\n") == 0){
        return -1;
    }
    if (strchr(buf, '&') != NULL){
        bg = 1;
        strcpy(buf, strtok(buf, "&"));
    }
    char* tok;
    tok = strtok(buf, " \n");
    argv[i] = tok;
    while (argv[i] != NULL){
        tok = strtok(NULL, " \n");
        argv[++i] = tok;
    }
    if (argv[0] == NULL){
        bg = -1;
    }
    return bg;
}

void do_cd (char **argv, char *path){
    if (argv[2] != NULL){
        fprintf(stderr, "Error: Too many arguments to cd. %s.\n", strerror(errno));
        return;
    }
    if (strcmp(argv[1], "~") == 0){
        struct passwd *password;
        if ((password = getpwuid(getuid())) != NULL){
            strcpy(argv[1], password->pw_dir);
            chdir(argv[1]);
        } else {
            fprintf(stderr, "Error: Cannot get passwd entry. %s.\n", strerror(errno));
        }
    } else if (strchr(argv[1], '~') != NULL){
        fprintf(stderr, "Error: Cannot access this directory with '~' in path. %s.\n", strerror(errno));
        return;
    } else if (chdir(argv[1]) == -1){
        fprintf(stderr, "Error: Cannot change directory to %s. %s.\n", argv[1], strerror(errno));
        return;
    }
   getcwd(path, 256);
   return;
}

char* getUser(){
    struct passwd *pws;
    uid_t uid = getuid();
    pws = getpwuid(uid);
    return pws->pw_name;
}

void eval(char* cmdline, char* cwd){
    char* argv[MAXARGS];
    char buf[MAXLINE];
    int bg;
    pid_t pid;
    strcpy(buf, cmdline);
    bg = parseline(buf, argv);
    if (bg == -1){
        return;
    }
    if (strcmp("cd", argv[0]) == 0){
        do_cd(argv, cwd);
        return;
    } else if (strcmp("exit", argv[0]) == 0){
        if (bg == 1){
            fprintf(stderr, "Error: Cannot run exit in background. %s.\n", strerror(errno));
            return;
        } else if (argv[1] != NULL){
            fprintf(stderr, "Error: Too many arguments. %s.\n", strerror(errno));
            return;
        }
        killpg(getpid(), SIGTERM);
		exit(0);
    } else if ((pid = fork()) == 0){
        execvp(argv[0], argv);
        fprintf(stderr, "Error: Invalid command %s. %s.\n", argv[0], strerror(errno));
    } else if (pid < 0) {
        fprintf(stderr, "Error: fork() failed. %s.\n", strerror(errno));
    }
    if (bg==0){
        int status;
        if (waitpid(pid, &status, 0) < 0){
            fprintf(stderr, "Error: wait() failed. %s.\n", strerror(errno));
        }
    } else {
        printf("pid: %d cmd: %s\n", pid, strtok(cmdline, "&"));
    }
    return;
}