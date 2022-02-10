import numpy as np
import pickle

class tictactoe_game():

    def game_is_over(self,game_data):
        if 0 in game_data:
            print("")
        else:
            print("Game tied")
            return True
        if self.is_winner(game_data)==True:
            return True
        else:
            return False
            
            
    def is_winner(self,board):
        if (board[0]==+1 and board[1]== +1 and board[2]==+1 ) or (board[3]==+1 and board[4]==+1 and board[5]==+1 ) or (board[6]==+1 and board[7]==+1 and board[8]==+1 ) or(board[0]==+1 and board[3]==+1 and board[6]== +1 ) or (board[1]==+1 and board[4]==+1 and board[7]==+1 ) or(board[2]==+1 and board[5]==+1 and board[8]==+1 ) or (board[0]==+1 and board[4]==+1 and board[8]==+1 ) or(board[2]==+1 and board[4]==+1 and board[6]==+1 ):
            print('Human player wins')
            return True
        elif (board[0]==-1 and board[1]== -1 and board[2]==-1 ) or (board[3]==-1 and board[4]==-1 and board[5]==-1 ) or (board[6]==-1 and board[7]==-1 and board[8]==-1 ) or(board[0]==-1 and board[3]==-1 and board[6]== -1 ) or (board[1]==-1 and board[4]==-1 and board[7]==-1 ) or(board[2]==-1 and board[5]==-1 and board[8]==-1 ) or(board[0]==-1 and board[4]==-1 and board[8]==-1 ) or(board[2]==-1 and board[4]==-1 and board[6]==-1 ):
            print('Computer player wins')
            return True
        else:
            return False
                
    def humanplayersturn(self,game_data,game_display):
        choice=int(input("Enter your choice:"))
        row=int(int(choice)/3)
        column=int(int(choice)%3)
        if game_display[row][column]==' ':
            game_data[choice]=+1
            game_display[row][column]='X'
            print(game_display)
            return game_data,game_display
        else:
            print('Already occupied')
            
            
    def computeraiturn(self,game_data,game_display):
        ai_choice=int(self.predict_ai(game_data))
        print("Computer AI choosed "+str(ai_choice))
        row_ai=int(ai_choice/3)
        column_ai=int(ai_choice%3)
        if game_display[row_ai][column_ai]==' ':
            game_data[ai_choice]=-1
            game_display[row_ai][column_ai]='O'
            print(game_display)
            return game_data,game_display
        else:
            print('Already occupied')
            
    def predict_ai(self,game_data):
        predict_output=[]
        for i in range(9):
            file_name = 'Model_param_col_'+str(i)+'.pkl'
            with open(file_name, 'rb') as f:
                tictactoe_ai_player = pickle.load(f)
            output=tictactoe_ai_player.predict(np.array(game_data).reshape(1, -1))
            predict_output.append(output)
        for i in range(9):
            if game_data[i]!=0:
                predict_output[i]=-1
        return predict_output.index(max(predict_output))    
           
    
if __name__ == '__main__':
    obj = tictactoe_game()

    game_display=np.array([
    [' ',' ',' '],
    [' ',' ',' '],
    [' ',' ',' ']
    ])

    game_data=[0 for i in range(9)]

    print("Human play with 'X' and the Computer AI plays with 'O'")

    first_player = input("Do you want to go first?? [Y/N]: ")

    if first_player=='Y':
        while(obj.game_is_over(game_data)!=True):
            game_data,game_display=obj.humanplayersturn(game_data,game_display)
            if(obj.game_is_over(game_data)!=True):
                game_data,game_display=obj.computeraiturn(game_data,game_display)
            else:
                break
    elif first_player=='N':
        while(obj.game_is_over(game_data)!=True):
            game_data,game_display=obj.computeraiturn(game_data,game_display)
            if(obj.game_is_over(game_data)!=True):
                game_data,game_display=obj.humanplayersturn(game_data,game_display)
            else:
                break	
    else:
        print('Wrong input entered')