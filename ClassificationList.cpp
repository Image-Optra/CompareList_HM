/**
 *  @file  ClassificationList.cpp
 *
 *  @brief  Implementation of the ClassificationList class.
 *
 *  Implementation of the ClassificationList class.
 *
 *  Copyright &copy; 2014  -  IRIS International, Inc.  -  All rights reserved
 */

  #include "ClassificationList.h"

  #include <boost/algorithm/string.hpp>
  #include <boost/lexical_cast.hpp>

  #include <iostream>


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

/**
 *  Constructs a ClassificationList from the given input stream (acl/pcl file).
 *
 *  @param [in]  stream  the input stream
 */

  APRT::ClassificationList::ClassificationList(std::istream& stream)
    {
      uint32_t ssn = 0;
//
//  Parse each <CLASS> line in the stream ...
//
      std::string input;
      while (std::getline(stream,input,'>'))
        {
          boost::trim_left(input);
          if (input == "<CLASS")  // the terminating > is discarded
            {
              this->classifications.push_back
                (ClassificationList::SubsampleClassifications(stream,++ssn));
            }
          else
            {
              std::getline(stream,input);  // discard the rest of the line
            }
        }
    }


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

/**
 *  Reads the classifications for the particles in a subsample.
 *
 *  @param [in]  stream  the input stream
 *  @param [in]  ssn     the subsample number
 *
 *  @return  the particle classifications
 */

  std::vector<APRT::PatchClassification>
    APRT::ClassificationList::SubsampleClassifications(std::istream&  stream,
                                                       const uint32_t ssn)
      {
        std::vector<PatchClassification> result;
        const std::istream_iterator<char> end;
              std::istream_iterator<char> iter(stream);
        std::string className;
        uint32_t index = 0;

        while (iter != end)
          {
            char nextChar = *iter++;
            if (nextChar == ','  ||
                nextChar == '<')
              {
                if (!className.empty())
                  {
                    result.push_back(PatchClassification(ssn,
                                                         index,
                                                         className));
                    className.clear();
                  }
                else
                  {
                    result.push_back(PatchClassification(ssn,
                                                         index,
                                                         "NONE"));
                  }
                ++index;
                if (nextChar == '<')
                  {
                    break;
                  }
              }
            else
              {
                className.push_back(nextChar);
              }
          }

        return (result);
      }
