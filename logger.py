def print_signature():
    llasm = """\
                               __    __       __         
                              / /   / /  __ _/ _\  /\/\  
                             / /   / /  / _` \ \  /    \ 
                            / /___/ /__| (_| |\ \/ /\/\ \\
                            \____/\____/\__,_\__/\/    \/
                             """

    logo = """\
                       __ _       _     __             _ 
                      / /(_)_ __ | | __/ _\ ___  _   _| |
                     / / | | '_ \| |/ /\ \ / _ \| | | | |
                    / /__| | | | |   < _\ \ (_) | |_| | |
                    \____/_|_| |_|_|\_\\\__/\___/ \__,_|_|
                                                         """

    print ("="*80)
    print (llasm)
    print (logo)
    print ("-"*80)
    print ("Demo/HuggingFace: https://huggingface.co/spaces/LinkSoul/LLaSM")
    print ("欢迎点一点 Star ^_^")
    print ("="*80)


if __name__ == '__main__':
    print_signature()