#include<stdio.h>
#include<ctype.h>
#include<string.h>
#include <math.h>
#include <stdlib.h>
#include<stdbool.h>
//////////////////////////////////////////////////
char board[3][3];
void boardinitialization() {
	char n='1';
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) {

			board[i][j]=n;
			n++;


		}
	}
}
void printboard() {
	printf("| %c | %c | %c |\n", board[0][0], board[0][1], board[0][2]);
	printf("|---|---|---|\n");
	printf("| %c | %c | %c |\n", board[1][0], board[1][1], board[1][2]);
	printf("|---|---|---|\n");
	printf("| %c | %c | %c |\n", board[2][0], board[2][1], board[2][2]);
	printf("|---|---|---|\n");


}
//to check winner
bool checkwinner(char player) {


	// Check rows
	// Check columns
	for (int i = 0; i < 3; i++) {
		if ((board[i][0] == player && board[i][1] == player && board[i][2] == player) ||
		        (board[0][i] == player && board[1][i] == player && board[2][i] == player)) {
			return true;
		}
	}

	// Check diagonals
	if ((board[0][0] == board[1][1] && board[1][1] == board[2][2] && board[0][0] == player) ||(board[0][2] == board[1][1] && board[1][1] == board[2][0] && board[0][2] == player)) {
		return true;
	}

	return false;
}
bool drawcase() {
	for (int i = 0; i < 3; i++) {
		for(int j=0; j<3; j++) {
			if(board[i][j]!='X'&&board[i][j]!='O') {
				return false;//not draw yet
			}
		}
	}
	return true;//draw!
}
//////////////////////////////////////////////////////


int minimax(bool ismaximizing) {
	//if ismaximizing then TRUE comp plays
	//if minimizing them ismaximing is FALSE player plays
	if(checkwinner('O')) {
		return 1;
	}
	if(checkwinner('X')) {
		return -1;
	}
	if(drawcase()) {
		return 0;
	}

	if(ismaximizing) { //TRUE=comp plays
		int bestscore=-1000;

		for(int i=0; i<3; i++) {
			for(int j=0; j<3; j++) {
				if(board[i][j]!='X'&&board[i][j]!='O') {
					char originalboard=board[i][j];
					board[i][j]='O';//do
					int score=minimax(false);//minimize
					board[i][j]=originalboard;//undo

					if(score>bestscore) {
						bestscore=score;

					}
				}


			}

		}
		return bestscore;
	}
	else { //ante FALSE player plays
		int bestscore=1000;
		for(int i=0; i<3; i++) {
			for(int j=0; j<3; j++) {
				if(board[i][j]!='X'&&board[i][j]!='O') {
					char originalboard=board[i][j];
					board[i][j]='X';//do
					int score=minimax(true);
					board[i][j]=originalboard;//undo


					if(score<bestscore) {
						bestscore=score;

					}
				}

			}


		}
		return bestscore;
	}
}

void computerplays() {
	int bestscore=-1000;
	int bestmoverow=-10;
	int bestmovecol=-10;
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) {
			if(board[i][j]!='X'&&board[i][j]!='O') {
				char originalboard=board[i][j];
				board[i][j]='O';//do
				int score=minimax(false);//minimize
				board[i][j]=originalboard;//undo


				if(score>bestscore) {
					bestscore=score;
					bestmoverow=i;
					bestmovecol=j;
				}
			}

		}
	}

	board[bestmoverow][bestmovecol] = 'O'; // Make the best move
	printf("computer's turn: \n");
	printboard();



}
///////////////////////////////////////////////////////////////
void playersturn() {
	int blockno;
	while(true) {
		printf("Enter your move: ");
		scanf("%d", &blockno);
		int row = (blockno - 1) / 3;
		int col = (blockno - 1) % 3;

		if (board[row][col] != 'X' && board[row][col] != 'O') {
			board[row][col] = 'X';
			printboard();
			break;

		}
	}
}
////////////////////////////////////////////////////////////////
int main() {
	boardinitialization();
	printboard();
	//game movement across player and comp
	printf("do you want to play first or second: ");
	char str[50];
	scanf("%s",str);
	if(strcmp(str,"first")==0) {
//player plays first
		int i =1;
		while(i<=9) {
			if(i%2!=0) {
				playersturn();
			}
			else {
				computerplays();
			}

			if (checkwinner('X')) {
				printf("You win!\n");
				break;
			}
			if (checkwinner('O')) {
				printf("Computer wins!\n");
				break;
			}
			if (drawcase()) {
				printf("It's a draw!\n");
				break;
			}
			i++;
		}

	}
	else if(strcmp(str,"second")==0) {
		int i=0;
		while(i <9) {
			if(i%2==0) {
				//even turns
				computerplays();
			}
			else {
				playersturn();
			}

			if (checkwinner('X')) {
				printf("You win!\n");
				break;
			}
			if (checkwinner('O')) {
				printf("Computer wins!\n");
				break;
			}
			if (drawcase()) {
				printf("It's a draw!\n");
				break;
			}
			i++;
		}
		return 0;
	}

}




