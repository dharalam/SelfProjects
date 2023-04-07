/*******************************************************************************
 * Name        : floyd.cpp
 * Author      : Dimitrios Haralampopoulos
 * Version     : 1.0
 * Date        : 12/07/2022
 * Description : Implementation of Floyd's Algorithm for All-Pairs-Shortest-Paths
 * Pledge      : I pledge my honor that I have abided by the Stevens Honor System
 ******************************************************************************/

#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>

using namespace std;

struct State{
    long long p; // path value
    long long i; // intermediate node

    State(){
        p = -1;
        i = -1;
    }

    State(long long _p, long long _i) :
        p{_p},i{_i} {}
};

int len(long a)
{
    int c = 0;
    while (a != 0)
    {
        a = a / 10;
        c++; 
    }
    return c;
}

void display_table(vector<vector<State>> matrix, const string &label, 
                   const bool use_letters = false) {
    if (!use_letters) {
    cout << label << endl; 
    long max_val = 1; 
    int num_vertices = matrix.size();
    for (int i = 0; i < num_vertices; i++) { 
        for (int j = 0; j < num_vertices; j++) { 
            long cell = matrix.at(i).at(j).p; 
            if (cell != -1 && cell > max_val) { 
                max_val = matrix.at(i).at(j).p; 
      } 
    } 
  } 
  int max_cell_width = use_letters ? len(max_val) : 
    len(max(static_cast<long>(num_vertices), max_val)); 
  cout << ' '; 
  for (int j = 0; j < num_vertices; j++) { 
    cout << setw(max_cell_width + 1) << static_cast<char>(j + 'A'); 
  } 
  cout << endl; 
  for (int i = 0; i < num_vertices; i++) { 
    cout << static_cast<char>(i + 'A'); 
    for (int j = 0; j < num_vertices; j++) { 
      cout << " " << setw(max_cell_width); 
      if (matrix.at(i).at(j).p == -1) { 
        cout << "-"; 
      }else { 
        cout << matrix.at(i).at(j).p; 
      } 
    } 
    cout << endl; 
  } 
  cout << endl; 
  } else {
    cout << label << endl; 
    long max_val = 1; 
    int num_vertices = matrix.size();
    for (int i = 0; i < num_vertices; i++) { 
        for (int j = 0; j < num_vertices; j++) { 
            long cell = matrix.at(i).at(j).i; 
            if (cell != -1 && cell > max_val) { 
                max_val = matrix.at(i).at(j).i; 
      } 
    } 
  } 
  int max_cell_width = use_letters ? len(max_val) : 
    len(max(static_cast<long>(num_vertices), max_val)); 
  cout << ' '; 
  for (int j = 0; j < num_vertices; j++) { 
    cout << setw(max_cell_width + 1) << static_cast<char>(j + 'A'); 
  } 
  cout << endl; 
  for (int i = 0; i < num_vertices; i++) { 
    cout << static_cast<char>(i + 'A'); 
    for (int j = 0; j < num_vertices; j++) { 
      cout << " " << setw(max_cell_width); 
      if (matrix.at(i).at(j).i == -1) { 
        cout << "-"; 
      } else{ 
        cout << static_cast<char>(matrix.at(i).at(j).i + 'A'); 
      }
    } 
    cout << endl; 
  } 
  cout << endl; 
  } 
  
} 


void printHelper(vector<vector<State>> inters, int current, int target)
{ 
    int grab = inters.at(current).at(target).i;
    if (grab == -1){
        cout << " -> " << static_cast<char>(target+65);
    } else {
        printHelper(inters, current, grab);
        printHelper(inters, grab, target);
        }
    
}

void printFloyds(vector<vector<State>> result)
{
    display_table(result,"Path lengths:");
    display_table(result,"Intermediate vertices:", true);
    
    for (size_t i = 0; i < result.size(); i++){
         for (size_t j = 0; j < result.size(); j++){
             cout << static_cast<char>(i+'A') << " -> " << static_cast<char>(j+'A') << ", ";
            if (result.at(i).at(j).p  != -1) {
             cout << "distance: " << result.at(i).at(j).p << ", path: " << static_cast<char>(i+65);
             if (!(i == j))
                 printHelper(result, i, j);
            }
            else {
                cout << "distance: infinity, path: none";
            }
            cout << "\n";
         }
     }
}

void floyds(int n, vector<vector<State>> distance){
    display_table(distance, "Distance matrix:");
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++){ 
                long long left;
                if (distance.at(i).at(k).p == -1 || distance.at(k).at(j).p == -1){
                    left = -1;
                } else {
                    left = distance.at(i).at(k).p + distance.at(k).at(j).p;
                }
                long long right = distance.at(i).at(j).p;
                if (left != -1 && (right == -1 || left < right)){
                    distance.at(i).at(j).p = left;
                    distance.at(i).at(j).i = k;
                }
            }
    printFloyds(distance);
}

vector<string> split (const string &s, char delim) {
    vector<string> result;
    stringstream ss (s);
    string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

int main(int argc, const char *argv[]) {
    // Make sure the right number of command line arguments exist.
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return 1;
    }
    // Create an ifstream object.
    ifstream input_file(argv[1]);
    // If it does not exist, print an error message.
    if (!input_file) {
        cerr << "Error: Cannot open file '" << argv[1] << "'." << endl;
        return 1;
    }
    // Add read errors to the list of exceptions the ifstream will handle.
    input_file.exceptions(ifstream::badbit);
    string line;
    int dim = 0;
    vector<vector<State>> matrix;
    try {
        unsigned int line_number = 1;
        // Use getline to read in a line.
        // See http://www.cplusplus.com/reference/string/string/getline/
        while (getline(input_file, line)) {
            string temp;
            if (line_number == 1){
                temp = line;
                if (!(('A'-temp[0])>0) || !(stoi(temp) >= 1 && stoi(temp) <= 26)){
                    cerr << "Error: Invalid number of vertices '" << temp << "' on line " << line_number << ".";
                    return 1;
                }
                dim = stoi(temp);
                vector<vector<State>> hmatrix = vector(dim, vector(dim, State()));  
                matrix = hmatrix;
                line_number++;
            } else {
                temp = line;
                vector<string> hold = split(temp, ' ');
                if (hold.size() != 3){
                    cerr << "Error: Invalid edge data '" << temp << "' on line " << line_number << ".";
                    return 1;
                }
                
                
                if (hold[0].length() > 1 || hold[0][0] < 'A' || hold[0][0] > (dim-1+'A')){
                    cerr << "Error: Starting vertex '" << hold[0] << "' on line " << line_number << " is not among valid values " << "A-" << static_cast<char>(dim-1 + 'A') << ".";
                    return 1;
                }
            
                if (hold[1].length() > 1 || hold[1][0] < 'A' || hold[1][0] > (dim-1+'A')){
                    cerr << "Error: Ending vertex '" << hold[1] << "' on line " << line_number << " is not among valid values " << "A-" << static_cast<char>(dim-1 + 'A') << ".";
                    return 1;
                }

                try{
                    if (stoi(hold[2]) < 1) {
                        cerr <<"Error: Invalid edge weight '" << hold[2] <<"' on line " << line_number << ".";
                        return 1;
                    }
                }
                catch(exception const & e){
                    cerr <<"Error: Invalid edge weight '" << hold[2] <<"' on line " << line_number << ".";
                    return 1;
                }

                
                matrix.at(hold[0][0]-'A').at(hold[1][0]-'A').p = stoi(hold[2]);
                line_number++;
                }
            }
        }
        catch (const ifstream::failure &f) {
        cerr << "Error: An I/O error occurred reading '" << argv[1] << "'.";
        return 1;
        }

        for (size_t i = 0; i < matrix.size(); i++){
            matrix.at(i).at(i).p = 0;
        }
        // Don't forget to close the file.
        input_file.close();
        floyds(matrix.size(),matrix);
        return 0;
    
}