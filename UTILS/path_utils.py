class manip():
    @staticmethod
    #-------------------------------------------------------------------------#
    def create_dir_if_not_exists(file_path):
        import os
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("directory created")
        else:
            print("directory exists")
    #-------------------------------------------------------------------------#
    @staticmethod
    def find_in_subdirs(source_directory, file_fix_glob_notation):
        '''
        files = manip.find_in_subdirs('D:\\path_to_dir\\', 'pre*') \n
        "files" contains each file name begins with 'pre'
        '''
        import glob
        files = []
        for filename in glob.iglob(source_directory + '/**/'+file_fix_glob_notation, recursive=True):
            files.append(filename)
        return files
    #-------------------------------------------------------------------------#    
    @staticmethod
    def copy_(source, destination):
        
        from shutil import copyfile
        
        copyfile(source, destination)
        