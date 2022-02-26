#include <iostream>
#include <memory>
#include <unordered_map>
// #include <map>
#include <string>

#include <bits/stdc++.h>

using namespace std;

int main() {
  int square_dim = 5000;
  
  struct coord {
    int x;
    int y;

    bool operator<(const coord &other) const {
        if (x < other.x) return true;
        if (other.x < x) return false;
	return (y < other.y);
    }

    bool operator==(const coord& p) const {
      return x == p.x && y == p.y;
    }
  };

  class MyHashFunction {
  public:
    // A hash function used to hash a pair of any kind
    size_t operator()(const coord& p) const {
      return ((p.x + p.y)*(p.x + p.y + 1) / 2) + p.y;
    }
  };


  const int QT_NODE_CAPACITY = 4;
  
  class quad_tree {
  public:
    quad_tree(int x0, int x1, int y0, int y1) {
      boundary_x0 = x0;
      boundary_x1 = x1;
      boundary_y0 = y0;
      boundary_y1 = y1;
    }
    
    int boundary_x0;
    int boundary_x1;
    int boundary_y0;
    int boundary_y1;

    pair<coord, string> points[QT_NODE_CAPACITY];
    int points_set = 0;
    
    shared_ptr<quad_tree> nw;
    shared_ptr<quad_tree> ne;
    shared_ptr<quad_tree> sw;
    shared_ptr<quad_tree> se;

    shared_ptr<string> lookup(coord p) {
      if (!nw) {
        for (unsigned int i = 0; i < QT_NODE_CAPACITY; i++) {
	  pair<coord, string> pt = points[i];
	  if (p == pt.first) {
	    return make_shared<string>(pt.second);
	  }
	}

	return nullptr;
      }
      
      if (((nw->boundary_x0 < p.x && p.x < nw->boundary_x1) &&
	   (nw->boundary_y0 < p.y && p.y < nw->boundary_y1))) {
	shared_ptr<string> ptr = nw->lookup(p);
	if (ptr) {
	  return ptr;
	}
      }
      
      if (((ne->boundary_x0 < p.x && p.x < ne->boundary_x1) &&
	    (ne->boundary_y0 < p.y && p.y < ne->boundary_y1))) {
	shared_ptr<string> ptr = ne->lookup(p);
	if (ptr) {
	  return ptr;
	}
      }
      
      if (((sw->boundary_x0 < p.x && p.x < sw->boundary_x1) &&
	    (sw->boundary_y0 < p.y && p.y < sw->boundary_y1))) {
        shared_ptr<string> ptr = sw->lookup(p);
	if (ptr) {
	  return ptr;
	}
      }
      
      if (((se->boundary_x0 < p.x && p.x < se->boundary_x1) &&
	    (se->boundary_y0 < p.y && p.y < se->boundary_y1))) {
        shared_ptr<string> ptr = se->lookup(p);
	if (ptr) {
	  return ptr;
	}
      }

      return nullptr;
    }
    
    bool insert(coord p, string s) {
      if (!((boundary_x0 < p.x && p.x < boundary_x1) &&
	    (boundary_y0 < p.y && p.y < boundary_y1))) {
	return false;
      }

      if (points_set < QT_NODE_CAPACITY && !nw) {
	points[points_set] = make_pair(p, s);
	points_set++;
	return true;
      }

      if (!nw) {
	// subdivide();

	int bm_x = (boundary_x0 + boundary_x1) / 2;
	int bm_y = (boundary_y0 + boundary_y1) / 2;
	
	nw = make_shared<quad_tree>(boundary_x0, bm_x, bm_y, boundary_y1);
	ne = make_shared<quad_tree>(bm_x, boundary_x1, bm_y, boundary_y1);
	sw = make_shared<quad_tree>(boundary_x0, bm_x, boundary_y0, bm_y);
	se = make_shared<quad_tree>(bm_x, boundary_x1, boundary_y0, bm_y);

	for (auto pt : points) {
	  if (nw->insert(pt.first, pt.second)) continue;
	  if (ne->insert(pt.first, pt.second)) continue;
	  if (sw->insert(pt.first, pt.second)) continue;
	  if (se->insert(pt.first, pt.second)) continue;
	}
      }

      if (nw->insert(p,s)) return true;
      if (ne->insert(p,s)) return true;
      if (sw->insert(p,s)) return true;
      if (se->insert(p,s)) return true;

      return false;
    }
  };
  
  
  // unordered_map<int, unordered_map<int, string>> double_map = unordered_map<int, unordered_map<int, string>>();
  // unordered_map<string, string> double_map = unordered_map<string, string>();

  // unordered_map<coord, string, MyHashFunction> double_map =
  //   unordered_map<coord, string, MyHashFunction>();

  // string double_map [2 * square_dim][2 * square_dim];

  // double_map.reserve(square_dim);
  // double_map.reserve(square_dim);

  quad_tree double_map(-square_dim, square_dim, -square_dim, square_dim);

  for (int x = -square_dim; x < square_dim; x++) {
    // double_map[x].reserve(square_dim);
    for (int y = -square_dim; y < square_dim; y++) {
      // double_map[x + square_dim][y + square_dim] = "test";
      // double_map[to_string(x + square_dim) + "," + to_string(y + square_dim)] = "test";
      // double_map[{x + square_dim, y + square_dim}] = "test";

      double_map.insert({x + square_dim, y + square_dim}, "test");
    }
  }

  for (int x = -square_dim; x < square_dim; x++) {
    for (int y = -square_dim; y < square_dim; y++) {
      // double_map[x + square_dim][y + square_dim];
      double_map.lookup({x + square_dim, y + square_dim});
    }
  }

  
  return 0;
}
