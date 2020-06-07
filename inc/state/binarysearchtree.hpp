#pragma once

namespace vuda
{
    namespace detail
    {
        /*
        binary search tree
        */

        /*
            Node is implemented with
                The Curiously Recurring Template Pattern (CRTP)
        */
        template<class Derived, typename KeyType>
        class bst_node
        {
        public:
            bst_node() :
                m_p(nullptr),
                m_left(nullptr),
                m_right(nullptr)
            {
            }

            virtual std::ostringstream print(int depth = 0) const = 0;

            //
            // sets
            inline void set_parent(Derived* node) { m_p = node; }
            inline void set_left(Derived* node) { m_left = node; }
            inline void set_right(Derived* node) { m_right = node; }
            inline void set_key(const KeyType& key, const size_t& range) { m_key = key; m_range = range; }
            //virtual void set_range(const size_t& range) { m_range = range; }
            //virtual void set_data(Derived* node) = 0;

            //
            // gets
            inline Derived* parent(void) const { return m_p; }
            inline Derived* left(void) const { return m_left; }
            inline Derived* right(void) const { return m_right; }
            inline KeyType key(void) const { return m_key; }
            inline size_t range(void) const { return m_range; }

        private:

            //
            // key
            KeyType m_key;
            size_t m_range;

        private:

            //
            // binary tree
            Derived* m_p;
            Derived* m_left;
            Derived* m_right;
        };

        template<class NodeType, typename KeyType>
        class bst
        {
        public:

            // inorder tree walk
            void walk(const NodeType* node)
            {
                if(node != nullptr)
                {
                    walk(node->left());
                    std::cout << node->print().str();
                    walk(node->right());
                }
            }

            void walk_depth(const NodeType* node, int depth=0)
            {
                if(node != nullptr)
                {
                    walk_depth(node->left(), depth + 1);
                    std::cout << node->print(depth).str();
                    walk_depth(node->right(), depth + 1);
                }
            }

            /*
            the operations
            - insert
            - delete
            run in O(h) on the bst of height h
            */

            // tree insert
            // note the current root can be changed to z
            void insert_node(NodeType*& root, NodeType* z)
            {
                NodeType* y = nullptr; // pointer to the parent of x
                NodeType* x = root; // pointer that traces the path downwards
                while(x != nullptr)
                {
                    y = x;

                    if(z->key() < x->key())
                        x = x->left();
                    else
                        x = x->right();
                }
                z->set_parent(y);
                if(y == nullptr)
                    root = z; // the tree was empty
                else
                {
                    if(z->key() < y->key())
                        y->set_left(z);
                    else
                        y->set_right(z);
                }
            }

            // tree delete
            // (a) if z has no children it is simply removed
            // (b) if z has one child. we splice z out
            // (c) if z has two children, we splice out its successor y (which has at most one child), and replace z's key and satelitete data with y's key and satelite data
            NodeType* delete_node(NodeType*& root, NodeType* z)
            {
                NodeType* x;
                NodeType* y;

                //
                // find the node y that is going to be spliced out
                // 1. either it is the input node z (if it has at most one child)
                // 2. or it is the successor of z (if it has two children)
                if(z->left() == nullptr || z->right() == nullptr)
                    y = z;
                else
                    y = successor(z);

                //
                // x is set to an existing child of y (or null if it has none).
                if(y->left() != nullptr)
                    x = y->left();
                else
                    x = y->right();

                
                //
                // the node y is going to be spliced out
                // 1. by modifying the parent of y
                // 2. and x
                // special cases are:
                // 1. y == root, i.e. the parent is null
                // 2. x == null, 
                
                if(x != nullptr)
                    x->set_parent(y->parent());

                if(y->parent() == nullptr)
                {
                    root = x;
                }
                else
                {
                    if(y == y->parent()->left())
                        y->parent()->set_left(x);
                    else
                        y->parent()->set_right(x);
                }

                //
                // if the successor is the one being spliced out, y should replace z.
                if(y != z)
                {
                    // y is moved into node z
                    //z->set_key(y->key(), y->range());
                    //z->set_data(y);

                    // we make y replace z instead of copying y's key and data
                    y->set_parent(z->parent());
                    y->set_left(z->left());
                    y->set_right(z->right());

                    //
                    // update the parent
                    if(y->parent() == nullptr)
                        root = y;
                    else
                    {
                        if(z == y->parent()->left())
                            y->parent()->set_left(y);
                        else
                            y->parent()->set_right(y);
                    }

                    //
                    // update the childrens parent, both children may not exist at this point
                    if(y->left() != nullptr)
                        y->left()->set_parent(y);
                    //else
                        //std::cout << "no left child" << std::endl;
                    if(y->right() != nullptr)
                        y->right()->set_parent(y);
                    //else
                        //std::cout << "no right child" << std::endl;

                    return z;
                }

                return y;
            }

            /*
                the operations
                 - search
                 - minimum
                 - maximum
                 - successor
                 - predecessor
                 run in O(h) on the bst of height h
            */

            /*// tree search recursive
            const bst_node* search_recursive(const bst_node* node, const KeyType key)
            {
                if(node == nullptr || key == node->key())
                    return node;

                if(key < node->key())
                    return search(node->left(), key);
                else
                    return search(node->right(), key);
            }*/

            // tree search iteratively
            NodeType* search(NodeType* node, const KeyType key) const
            {
                while(node != nullptr && key != node->key())
                {
                    if(key < node->key())
                        node = node->left();
                    else
                        node = node->right();
                }
                return node;
            }

            // tree search iteratively
            NodeType* search_range(NodeType* node, const KeyType key) const
            {
                //
                // NOTE - ptrdiff_t may not be large enough to hold the difference between memory adresses.
                //std::ptrdiff_t diffkey;
                //std::size_t mindiffkey = std::numeric_limits<std::size_t>::max(); // std::numeric_limits<std::ptrdiff_t>::max();

                std::int8_t diffkey_sign;
                std::size_t absdiffkey;
                std::int8_t minkey_sign = 1;
                std::size_t minkey = (std::numeric_limits<std::size_t>::max)();
                NodeType* minnode = nullptr;
            
                //
                // find the closest node wrt. key
                while(node != nullptr)
                {
                    //
                    // store the difference between the keys with full precision
                    if(key < node->key())
                    {
                        // key is to the left of the node key
                        absdiffkey = static_cast<const char*>(node->key()) - static_cast<const char*>(key);
                        diffkey_sign = -1;
                    }
                    else
                    {
                        // key is to the right of the node key
                        absdiffkey = static_cast<const char*>(key) - static_cast<const char*>(node->key());
                        diffkey_sign = 1;
                    }
                    
                    //
                    // return the node if the key is an exact match
                    if(absdiffkey == 0)
                        return node;

                    //
                    // store absolute minimum distance and its sign
                    if(absdiffkey < minkey)
                    {
                        minkey_sign = diffkey_sign;
                        minkey = absdiffkey;
                        minnode = node;
                    }

                    if(key < node->key())
                        node = node->left();
                    else
                        node = node->right();
                }

                //
                // when minkey is negative we find the predecessor
                if(minkey_sign < 0)
                {
                    //std::cout << "return successor: " << mindiffkey << std::endl;
                    node = predecessor(minnode);

                    //
                    // this is left most node (minimum node) - the key is not in range of any node!
                    if(node == nullptr)
                        return nullptr;
                }
                else
                {
                    //
                    // else the key is to the right and we keep the minnode
                    node = minnode;
                    assert(node);
                }

                //
                // now that we have the closest node and the key is ensured to be to the right of the node (minkey_sign = 1)
                // we can check if we are out of bound 
                absdiffkey = static_cast<const char*>(key) - static_cast<const char*>(node->key());
                
                //assert(diffkey >= 0); // absdiffkey is ensured to be positive
                if(absdiffkey < minnode->range())
                {
                    //std::cout << "return minnode: " << mindiffkey << std::endl;
                    return node;
                }
                else
                    return nullptr;
            }

            // tree minimum
            NodeType* minimum(NodeType* node) const
            {
                while(node->left() != nullptr)
                    node = node->left();
                return node;
            }

            // tree maximum
            NodeType* maximum(NodeType* node) const
            {
                while(node->right() != nullptr)
                    node = node->right();
                return node;
            }

            // tree successor
            // returns the successor of a node x in the bst if it exists, and nullptr if the node has the largest key in the tree
            NodeType* successor(NodeType* node) const
            {
                if(node->right() != nullptr)
                    return minimum(node->right());

                NodeType* suc = node->parent();

                while(suc != nullptr && node == suc->right())
                {
                    node = suc;
                    suc = suc->parent();
                }
                return suc;
            }

            // tree predecessor
            NodeType* predecessor(NodeType* node) const
            {
                if(node->left() != nullptr)
                    return maximum(node->left());

                NodeType* pre = node->parent();

                while(pre != nullptr && node == pre->left())
                {
                    node = pre;
                    pre = pre->parent();
                }
                return pre;
            }

        private:
        
        };

    } //namespace detail
} //namespace vuda
