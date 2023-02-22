#include <iostream>
#include <cmath>


using namespace std;



class FastMCIntegral {
    int width, height;
  public:
    void set_values (int,int);
    int area() {return width*height;}
};

void FastMCIntegral::set_values (int x, int y) {
  width = x;
  height = y;
}


void FastMCIntegral::get_contents() {


}



void FastMCIntegral::update_values() {

}