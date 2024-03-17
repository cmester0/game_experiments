#include <iostream>
#include <stdio.h>

using namespace std;

int main() {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen("echo '16 27\n0 1\n0 2\n0 6\n1 3\n1 5\n1 7\n2 3\n2 4\n3 5\n4 5\n4 6\n5 7\n6 8\n8 9\n8 10\n8 11\n9 10\n9 11\n10 11\n7 13\n7 14\n9 12\n9 15\n12 13\n12 15\n13 14\n14 15' | python ../cheat/cheat.py", "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);

    cout << result << endl;
    
    return 0;
}
