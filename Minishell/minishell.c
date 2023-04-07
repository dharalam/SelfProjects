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
char cwd[PATH_MAX];
#define MAXARGS (256)
#define MAXLINE (256)

int main(){
    char cmdline[MAXLINE];
    shinstaller();
    getcwd(cwd, 256);
    while (1){
        printf("minishell:%s:%s$ ", getUser(), cwd);
        fgets(cmdline, MAXLINE, stdin);
        if (feof(stdin)){
            exit(0);
        }
        eval(cmdline, cwd);
    }

}
