#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <typeinfo>
#include <bits/stdc++.h>
#include <bitset>

#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>

#include <unistd.h>
#include <fcntl.h>
using namespace std;

float FLIP_RATIO = 0.02;
int counter = 0;
int n=0;
vector<int> status_codes;

vector<vector<char>> magic_vals =   
{
	{char(0xFF)},
  	{char(0x7F)},
  	{char(0x00)},
  	{char(0xFF),char(0xFF)},
  	{char(0x00),char(0x00)},
  	{char(0xFF),char(0xFF), char(0xFF), char(0xFF)},
  	{char(0x00),char(0x00), char(0x00), char(0x00)},
  	{char(0x00),char(0x00), char(0x00), char(0x80)},
  	{char(0x00),char(0x00), char(0x00), char(0x40)},
  	{char(0xFF),char(0xFF), char(0xFF), char(0x7F)}
 };

vector<int> flip_array = {1, 2, 4, 8, 16, 32, 64, 128};

std::vector<char> read_jpg(){

    std::ifstream input( "deneme.jpg" );
	std::ofstream output( "deneme2.jpg" );
    
    std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
    
    return buffer;
}

void copyDataset(int i)
{
    std::ifstream  src("output.jpg", std::ios::binary);
    std::ofstream  dst("dataset/"+to_string(i)+"_output.jpg",   std::ios::binary);

    dst << src.rdbuf();
}

vector<char> writeFile(vector<char> data)
{
	std::ofstream file;
	file.open("output.jpg", std::ios_base::binary);
    for(int i = 0; i < data.size(); ++i){
       file.write((char*)(&data[i]), sizeof(data[i]));
    }
    file.close();
}

vector<char> readFile()
{
	ifstream myfile("tmp.jpg", ios::binary | ios::in);
	string line;
	vector<char> lines;
	//myfile.open ("deneme.jpg");
	char c;
	while ( myfile.get(c) )
    {
	    for (int i = 7; i >= 0; i--){	    	
	        c >> i;    	
	    }
    	lines.push_back(c);
    }
	myfile.close();
	return lines;
}

char bit_flip(char flip_it){
	int choice = rand()%flip_array.size();
	char flipped = flip_it ^ flip_array[choice];
	return flipped;
}

vector<char> magic(vector<char> data, int idx){


  	int choice = rand()%magic_vals.size();

  	int offset=0;
  	for(int i=0;i<magic_vals[i].size();i++){
  		data[idx+offset] = magic_vals[i][offset];
  		offset+=1;
  	}
  	return data;
}

vector<char> mutate(vector<char> data,float flips){
	vector<int> random_indexes;
	for(int i=0;i<flips;i++){
		random_indexes.push_back(rand()%(data.size() - 8)+2);
	} 
	int methods[] = {0,1};
//	cout << typeid(a).name() << endl;
	for(int i=0;i<random_indexes.size();i++){
		int method = rand()%2;
		if(method==0){


			data[random_indexes[i]]=bit_flip(data[random_indexes[i]]);
		}else{
			data = magic(data,random_indexes[i]);
		}
	}

	return data;
}

void sig_handler(int signum){
	counter+=1; 
	n+=1;
	copyDataset(n);
}

void run_child(){
/*
	char* cmd[] = { "file", "/home/x/security/fuzzer/output.jpg", NULL };
	pid_t child;
	child = fork();
	if(child==0){
		execvp(cmd[0],cmd);
		exit(3);
	}else{

	}
*/

	/*if(execl("/home/x/security/fuzzer/output.jpg","file")!=0){
		counter+=1;
	}*/
	int p[2];
	size_t size = sizeof(int);
	pid_t child;
	
	char *args[]={"jpeg2hdf","output.jpg","hh.hdf",NULL};
//	char *args[]={"./fuzz_this","output.jpg",NULL};
	child = fork();	
	if(child==0) {
		//ptrace(PTRACE_TRACEME, 0, NULL, NULL);
	    int fd = open("/dev/null", O_WRONLY);

	    //dup2(fd, 1);    /* make stdout a copy of fd (> /dev/null) */
	    //dup2(fd, 2);    /* ...and same with stderr */
	    //close(fd);      /* close fd */
			alarm(3);		
		if(execvp(args[0],args)!=0){
			//cout << "1111" << endl;
			//ofstream myfile;
			//myfile.open ("example.txt");
			//myfile << "Writing this to a file.\n";
			//myfile.close();
			//counter = counter + 1;
			//write(p[0], &counter, sizeof(counter));
		}else{
			//int n=0;
			//write(p[0], &n, sizeof(counter));			
		}
		exit(0);
	} else {
		//int n=0;
		//read(p[1],&n,sizeof(n));
		//cout << n << endl;
		int status;
			signal(SIGALRM,sig_handler); // Register signal handler
			alarm(2);
 	   	waitpid(child, &status, 0);

 	   	if(status!=0){
 	   		if(std::find(status_codes.begin(), status_codes.end(),status)==status_codes.end()){
	 	   		status_codes.push_back(status);
 	   		}
		 	//counter+=1;
 	   	}
        //ptrace(PTRACE_CONT, child, NULL, NULL);
    }
}

void print_codes(){
	for(int i=0;i<status_codes.size();i++){
		if(i==status_codes.size()-1){
			cout << status_codes[i] << endl;		
			break;
		}
		cout << status_codes[i] << ",";		
	}	
}

int main(int argc, char** argv){
	string filename = argv[1];
	cout << filename << endl;

	//std::vector<unsigned char> jpeg_file = read_jpg();
	std::vector<char> jpeg_file = readFile();
	vector<char> copy_data; 
	vector<char> mutated;
	float flips = jpeg_file.size()*FLIP_RATIO;
	for(int i=0;i<1000;i++){
		copy_data = jpeg_file;
		mutated = mutate(copy_data,flips);
			
		writeFile(mutated);
		run_child();
	}
	cout << "Number of crashes: " << counter << endl;
	cout << "Number of unique crashes: " << status_codes.size() << endl;
	cout << "Status codes are: " << endl;
	print_codes();
	return 1;
}