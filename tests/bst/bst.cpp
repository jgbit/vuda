#pragma once

#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#endif

#include <vuda.hpp>
#include <random>

int main(int argc, char *argv[])
{
    vuda::bst<vuda::bst_default_node, void*> storage;

    const int num = 52;
    vuda::bst_default_node bst[num];
    vuda::bst_default_node* root = nullptr;

    int rnd[num];

    //    
    // create tree

    for(int i = 0; i < num; ++i)
    {
        bst[i].set_key(&bst[i]);
        rnd[i] = i;
    }

    // BS size
    std::ptrdiff_t diffsize = abs(static_cast<char*>(bst[1].key()) - static_cast<char*>(bst[0].key()));

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, num - 1);
    std::uniform_int_distribution<int> dist2(1, num - 1);

    // shuffle
    for(int i = 0; i < num; ++i)
    {
        int v1 = dist(mt);
        int v2 = dist(mt);
        std::swap(rnd[v1], rnd[v2]);
    }

    for(int i = 0; i < num; ++i)
    {
        std::cout << rnd[i] << " ";
        storage.insert_node(root, &bst[rnd[i]]);
    }
    std::cout << std::endl;

    //
    // print tree
    std::cout << "tree: " << std::endl;
    //storage.walk(root);
    storage.walk_depth(root);

    //
    // delete a node
    storage.delete_node(root, &bst[0]);

    //
    // print tree
    std::cout << "tree after deletion: " << std::endl;
    //storage.walk(root);
    storage.walk_depth(root);

    //
    // operations    

    const vuda::bst_default_node* searchnode = nullptr;
    const int ref = dist2(mt);
    void* value0 = bst[ref].key();

    const int addptr = 6;
    //void* value1 = static_cast<char*>(value0) + addptr*diffsize - 17;
    //void* value1 = static_cast<char*>(value0) + addptr*diffsize + 13;
    //void* value1 = static_cast<char*>(value0) - addptr*diffsize - 13;
    void* value1 = static_cast<char*>(value0) - addptr * diffsize + 13;

    searchnode = storage.search(root, value0);
    std::cout << "search (" << value0 << "): ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.search(root, value1);
    std::cout << "search (" << value1 << "): ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    //
    // FUN WITH POINTERS
    /*int size = sizeof(void*);
    void* ptr1 = nullptr;
    //void* ptr2 = static_cast<char*>(ptr1) + 10;
    void* ptr2 = static_cast<char*>(ptr1) + (std::numeric_limits<uint64_t>::max)() / 2 + 1;

    // std::ptrdiff_t is the signed integer type of the result of subtracting two pointers.
    std::ptrdiff_t diff12 = static_cast<char*>(ptr1) - static_cast<char*>(ptr2);
    std::ptrdiff_t diff21 = static_cast<char*>(ptr2) - static_cast<char*>(ptr1);
    auto min = PTRDIFF_MIN;
    auto max = PTRDIFF_MAX;
    uint64_t ad2 = 2 * (PTRDIFF_MAX)+1;
    auto ad = (uint64_t)ptr2;*/

    searchnode = storage.search_range(root, value1);
    std::cout << "search range (" << value1 << "): ";
    if(searchnode)
    {
        searchnode->print();
        std::cout << "distance between elements: " << (static_cast<char*>(searchnode->key()) - static_cast<char*>(value0)) / diffsize << std::endl;
    }
    else
        std::cout << "NO NODE WAS RETURNED!" << std::endl;

    searchnode = storage.minimum(root);
    std::cout << "min: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.maximum(root);
    std::cout << "max: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.predecessor(root);
    std::cout << "predecessor: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.successor(root);
    std::cout << "successor: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;
}