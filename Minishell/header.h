// Author: Dimitrios Haralampopoulos
// I pledge my honor that I have abided by the Stevens Honor System

void chandler (int sig);
void shandler (int sig);
void shinstaller ();
int parseline (char *buf, char **argv);
void do_cd (char **argv, char *path);
char* getUser();
void eval(char* cmdline, char* cwd);

