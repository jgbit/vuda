#pragma once

namespace vuda
{

    /*
    placeholder for now
    we want something like

    specialization<int, int, float> info;
    info.set<0>(13);
    info.set<1>(24);
    info.set<2>(23.55f);

    might require c++17

    */
    struct specialization
    {
        uint32_t m_data;
        vk::SpecializationMapEntry m_mapEntry;
    };

} //namespace vuda