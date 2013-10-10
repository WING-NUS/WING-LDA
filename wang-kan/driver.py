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

    docs = 1                    # TODO: stub
    v = 1                       # TODO: stub

    lda = lda.LDA(docs,v)
    print ("hello")
# end main()

# execute class if called from command-line
if __name__ == "__main__":
    main()
