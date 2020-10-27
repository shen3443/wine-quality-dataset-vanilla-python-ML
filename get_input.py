class GetInput:
    '''
    Class that handles recieving user input
    '''

    def __init__(self):
        print("For more on your options, please see:")
        print(
            "https://github.com/shen3443/"
            "wine-quality-dataset-vanilla-python-ML"
            "/blob/main/README.md#settings"
            )
        print()

    def get_activation(self):
        '''
        Prompts user for input and returns a string representing
        the chosen activation function.
        '''

        # Give menu to user, prompt for selection
        print(
            "Please select an Activation Function for the hidden layer:"
            " \n1. Sigmoid \n2. ReLU\n3. leaky ReLU"
        )
        answer = input("Your selection: ")
        print()

        # Return the answer in an expected format
        if answer == '1' or answer == 'Sigmoid':
            return 'sigmoid'
        elif answer == '2' or answer == 'ReLU':
            return 'ReLU'
        elif answer == '3' or answer == 'leaky ReLU':
            return 'leaky ReLU'

        # If the input is not accepted, recursively
        # call the method to try again
        else:
            print(
                "Invalid response. Please enter the number next to the"
                " function on the menu or the name of the function exactly"
                " as it appears in the menu..."
                )
            return self.get_activation()

    def get_test_size(self):
        '''
        Prompts user for input and returns a float representing the portion of
        data to be reserved for testing
        '''

        # Explain to the user what they are being asked,
        # then prompt for response
        print(
            "Please enter the portion of data you would like to reserve for"
            " testing the model as a percentage between 1 and 99"
            )
        print(
            "**Data reserved for testing is not used to train the model. It"
            " is recommended that no more that 20 percent of the"
            " data be reserved"
            )
        answer = input("Percent reserved: ")
        print()

        # Remove % symbol if it is passed in the input
        if answer[-1] == '%':
            answer = answer[:-1]

        # Use try/except to handle possible inputs that
        # cannot be converted to floats
        try:
            answer = float(answer)

            # Check that the answer is within expected range
            if 0 < answer < 100:
                return answer / 100

            # If the input is out of the expected range,
            # recursively call the method to try again
            print(
                "Invalid response - answer out of range."
                " Please enter a number between 1 and 99..."
                )
            return self.get_test_size()

        except ValueError:
            # If the input cannot be converted to float,
            # recursively call the method to try again
            print(
                "Invalid response - unable to convert to float."
                " Please enter a number between 1 and 99..."
                )
            return self.get_test_size()

    def get_feature_scale_technique(self):
        '''
        Prompts user for input and returns a string representing
        the chosen feature scaling technique
        '''

        # Give menu to user, prompt for selection
        print(
            "Please select a scaling technique for the features (inputs):"
            " \n1. Standard \n2. Normal"
            )
        answer = input("Your selection: ")
        print()

        # Return the answer in an expected format
        if answer == '1' or answer == 'Standard':
            return 'standardize'
        elif answer == '2' or answer == 'Normal':
            return 'normalize'

        # If the input is not accepted, recursively
        # call the method to try again
        else:
            print(
                "Invalid response. Please enter the number next to the"
                " technique on the menu or the name of the technique"
                " exactly as it appears in the menu..."
                )
            return self.get_feature_scale_technique()
