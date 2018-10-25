#pragma once

namespace vuda
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

        virtual void print(int depth = 0) const {}

        //
        // sets
        void set_parent(Derived* node) { m_p = node; }
        void set_left(Derived* node) { m_left = node; }
        void set_right(Derived* node) { m_right = node; }
        virtual void set_key(const KeyType& key, const size_t& range) { m_key = key; m_range = range; }
        //virtual void set_range(const size_t& range) { m_range = range; }
        virtual void set_data(Derived* node) = 0;

        //
        // gets
        Derived* parent(void) const { return m_p; }
        Derived* left(void) const { return m_left; }
        Derived* right(void) const { return m_right; }
        KeyType key(void) const { return m_key; }
        size_t range(void) const { return m_range; }

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

    class bst_default_node : public bst_node<bst_default_node, void*>
    {
    public:

        bst_default_node()
        {
            set_key(this, 1);            
        }

        void print(int depth = 0) const
        {
            std::ostringstream ostr;
            for(int i = 0; i < depth; ++i)
                ostr << "-";
            ostr << key() << " " << (uintptr_t)key() << std::endl;
            std::cout << ostr.str();
        }

        void set_data(bst_default_node* node)
        {
            m_example = node->m_example;
        }

    private:

        // satellite data
        float m_example;
    };

    class bst_derived_node : public bst_default_node
    {
    public:

        void set_data(bst_default_node* node)
        {
            //
            // invoke base
            bst_default_node::set_data(node);

            // copy node's satellite data                        
            bst_derived_node* deriv = static_cast<bst_derived_node*>(node);
            m_moredata = deriv->m_moredata;
        }

    private:
        float m_moredata;
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
                node->print();
                walk(node->right());
            }
        }

        void walk_depth(const NodeType* node, int depth=0)
        {
            if(node != nullptr)
            {
                walk_depth(node->left(), depth + 1);
                node->print(depth);
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

            if(z->left() == nullptr || z->right() == nullptr)
                y = z;
            else
                y = successor(z);

            if(y->left() != nullptr)
                x = y->left();
            else
                x = y->right();

            if(x != nullptr)
                x->set_parent(y->parent());

            if(y->parent() == nullptr)
                root = x;
            else
            {
                if(y == y->parent()->left())
                    y->parent()->set_left(x);
                else
                    y->parent()->set_right(x);
            }

            if(y != z)
            {
                z->set_key(y->key(), y->range());
                z->set_data(y);
                // copy y's satelite data into z
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

            std::ptrdiff_t diffkey;
            std::ptrdiff_t mindiffkey = std::numeric_limits<std::ptrdiff_t>::max();
            std::size_t minkey = std::numeric_limits<std::size_t>::max();
            NodeType* minnode = nullptr;
            
            while(node != nullptr)
            {
                diffkey = static_cast<const char*>(key) - static_cast<const char*>(node->key());

                if(diffkey == 0)
                    return node;

                if((size_t)abs(diffkey) < minkey)
                {
                    mindiffkey = diffkey;
                    minkey = abs(diffkey);
                    minnode = node;
                }

                if(key < node->key())
                    node = node->left();
                else
                    node = node->right();
            }

            if(mindiffkey < 0)
            {
                //std::cout << "return successor: " << mindiffkey << std::endl;
                node = predecessor(minnode);

                if(node == nullptr) // this is left most node (minimum node)
                    return nullptr;

                //if(node != nullptr)
                    //int break_point = 1;
            }
            else
            {
                node = minnode;
            }

            //
            // check if we are out of bound
            diffkey = static_cast<const char*>(key) - static_cast<const char*>(node->key());
            assert(diffkey >= 0);
            if((size_t)abs(diffkey) < minnode->range())
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

} //namespace vuda
