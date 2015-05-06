/**
 *  @file  ClassificationList.h
 *
 *  @brief  Definition of the ClassificationList class.
 *
 *  Definition of the ClassificationList class.
 *
 *  Copyright &copy; 2014  -  IRIS International, Inc.  -  All rights reserved
 */

  #ifndef   APRT_CLASSIFICATION_LIST_H_INCLUDED
    #define APRT_CLASSIFICATION_LIST_H_INCLUDED

    #include <boost/tuple/tuple.hpp>

    #include <iosfwd>
    #include <string>
    #include <vector>

    #include <cassert>

    #include <stdint.h>


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

    namespace APRT
      {

/**
 *  The classification information for a single patch.
 */

        struct PatchClassification
          {
            PatchClassification(uint32_t           ssn,
                                uint32_t           idx,
                                const std::string& cls);
            uint32_t    subsampleNumber;  /**< @brief  one-based subsample number      */
            uint32_t    patchIndex;       /**< @brief  zero-based for each subsample   */
            std::string classification;   /**< @brief  the apr- or user-assigned class */
          };

        inline bool operator == (const PatchClassification& A,
                                 uint32_t                   B)
          {
            return (A.patchIndex == B);
          }

/**
 *  A container for apr and user classifications for the particles in
 *  a multiple subsample runfile.
 */

        class ClassificationList
          {
            public:
              ClassificationList();
              ClassificationList(std::istream& stream);

            public:
              const std::vector<std::vector<PatchClassification> >&
                Classifications() const;
            private:
              static std::vector<PatchClassification>
                SubsampleClassifications(std::istream& stream,
                                         uint32_t      ssn);
            private:
              std::vector<std::vector<PatchClassification> > classifications;
                /**< @brief  the classifications for the patches */
          };
      }


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

/**
 *  Creates a PatchClassification with the given values.
 */

    inline APRT::PatchClassification::PatchClassification(const uint32_t     ssn,
                                                          const uint32_t     idx,
                                                          const std::string& cls)
      : subsampleNumber(ssn),
        patchIndex(idx),
        classification(cls)
          {
            ;
          }


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

/**
 *  Creates an empty ClassificationList.
 */

    inline APRT::ClassificationList::ClassificationList()
      {
        ;
      }


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

/**
 *  Returns the classifications.
 *
 *  @return  the classifications
 */

    inline const std::vector<std::vector<APRT::PatchClassification> >&
      APRT::ClassificationList::Classifications() const
        {
          return (this->classifications);
        }

  #endif
