"""
Copyright 2013 by National University of Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy
def main():
    """Main function for command line invocation
    """
    import lda

    # Assuming that there are 3 topics in the corpus and following are the
    # distributions of terms in those topics:
    # Topic 1 : { 1, 4, 5, 7 }
    # Topic 2 : { 2, 6, 9 }
    # Topic 3 : { 3, 8, 0 }

    docs = [[1,2,4],[2,3,6],[1,8,0],[5,2,9],[0,8,5],  \
            [5,6,7],[8,9,0],[4,7,0],[0,3,2],[7,8,1],  \
            [1,5,2],[4,5,9],[6,9,0],[3,8,7],]
    v = 10                      # 10 vocabulary items

    lda = lda.LDA(docs,v)
# end main()

# execute class if called from command-line
if __name__ == "__main__":
    main()
