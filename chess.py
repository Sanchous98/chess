from __future__ import annotations

import os
from enum import IntFlag, auto
from typing import Optional, Union
import re


class MoveNotPossibleException(BaseException):
    pass


class InvalidCellException(BaseException):
    pass


class PieceType(IntFlag):
    WHITE = 0
    BLACK = 1

    def __str__(self):
        return 'white' if self == self.WHITE else 'black'

    def opposite(self):
        return PieceType(not self)


class Piece:
    def __init__(self, color: PieceType):
        if color not in (PieceType.BLACK, PieceType.WHITE):
            raise ValueError('invalid piece color')

        self.__color = color

    @property
    def color(self) -> PieceType:
        return self.__color

    def handle_event(self, board: Chess, event: Event):
        raise NotImplemented

    def assert_move(self, piece: Piece, board: Chess, event: Event):
        """ Base asserts, that are actual for all pieces"""

        """ Zero move is wrong """
        if event.start_pos[0] - event.end_pos[0] == 0 and event.start_pos[1] - event.end_pos[1] == 0:
            raise MoveNotPossibleException

        """ If a piece eats another, the cell must not be empty """
        if event.action == Event.EAT and board.get_piece(event.end_pos) is None:
            raise MoveNotPossibleException

        """ From the other side, if piece moves, the target cell must be free """
        if event.action == Event.MOVE and board.get_piece(event.end_pos) is not None:
            raise MoveNotPossibleException

        """ The cell should not be occupied by a friendly piece """
        if board.get_piece(event.end_pos) is not None and board.get_piece(event.end_pos).color == piece.color:
            raise MoveNotPossibleException

        """ Check for misuses. Actually current piece should be found in the same position """
        if event.start_pos != board.find_position(piece):
            raise MoveNotPossibleException


class Pawn(Piece):
    def handle_event(self, board: Chess, event: Event):
        self.assert_move(self, board, event)
        board.place_piece(self, event.end_pos)

    def assert_move(self, piece: Piece, board: Chess, event: Event):
        super().assert_move(piece, board, event)

        if event.action == event.MOVE:
            if event.start_pos[1] != 1 and event.start_pos[1] != 6:
                if abs(event.start_pos[1] - event.end_pos[1]) != 1:
                    raise MoveNotPossibleException
            else:
                if abs(event.start_pos[1] - event.end_pos[1]) not in (1, 2):
                    raise MoveNotPossibleException
        elif event.action == event.EAT:
            if piece.color == PieceType.WHITE:
                if event.end_pos[1] - event.start_pos[1] != 1:
                    raise MoveNotPossibleException
            else:
                if event.start_pos[1] - event.end_pos[1] != 1:
                    raise MoveNotPossibleException

            if abs(event.end_pos[0] - event.start_pos[0]) != 1:
                raise MoveNotPossibleException

    def __str__(self):
        return 'P'


class King(Piece):
    def handle_event(self, board: Chess, event: Event):
        self.assert_move(self, board, event)
        board.place_piece(self, event.end_pos)

    def assert_move(self, piece: Piece, board: Chess, event: Event):
        super().assert_move(piece, board, event)

        if abs(event.end_pos[0] - event.start_pos[0]) != 1:
            raise MoveNotPossibleException

        if abs(event.end_pos[1] - event.start_pos[1]) != 1:
            raise MoveNotPossibleException

        """ TODO: Check if not check """

    def __str__(self):
        return 'K'


class Knight(Piece):
    def handle_event(self, board: Chess, event: Event):
        self.assert_move(self, board, event)
        board.place_piece(self, event.end_pos)

    def assert_move(self, piece: Piece, board: Chess, event: Event):
        super().assert_move(piece, board, event)

        if abs(event.start_pos[0] - event.end_pos[0]) == 2:
            if abs(event.start_pos[1] - event.end_pos[1]) != 1:
                raise MoveNotPossibleException
        elif abs(event.start_pos[1] - event.end_pos[1]) == 2:
            if abs(event.start_pos[0] - event.end_pos[0]) != 1:
                raise MoveNotPossibleException

    def __str__(self):
        return 'N'


class Bishop(Piece):
    def handle_event(self, board: Chess, event: Event):
        self.assert_move(self, board, event)
        board.place_piece(self, event.end_pos)

    def assert_move(self, piece: Piece, board: Chess, event: Event):
        super().assert_move(piece, board, event)

        if abs(event.start_pos[0] - event.end_pos[0]) == abs(event.start_pos[1] - event.end_pos[1]):
            return

        raise MoveNotPossibleException

    def __str__(self):
        return 'B'


class Rook(Piece):
    def handle_event(self, board: Chess, event: Event):
        self.assert_move(self, board, event)
        board.place_piece(self, event.end_pos)

    def assert_move(self, piece: Piece, board: Chess, event: Event):
        super().assert_move(piece, board, event)

        if event.start_pos[0] != event.end_pos[0] and event.start_pos[1] != event.end_pos[1]:
            raise MoveNotPossibleException

        cols = list(range(event.start_pos[0], event.end_pos[0]))
        rows = list(range(event.start_pos[1], event.end_pos[1]))

        if len(rows) == 0:
            rows += [event.start_pos[1]] * len(cols)
        else:
            cols += [event.start_pos[0]] * len(rows)

        for pos in list(zip(cols, rows))[1:]:
            if board.get_piece(pos) is not None:
                raise MoveNotPossibleException

    def __str__(self):
        return 'R'


class Queen(Piece):
    def handle_event(self, board: Chess, event: Event):
        self.assert_move(self, board, event)
        board.place_piece(self, event.end_pos)

    def assert_move(self, piece: Piece, board: Chess, event: Event):
        super().assert_move(piece, board, event)

        if event.start_pos[0] != event.end_pos[0] and event.start_pos[1] != event.end_pos[1]:
            """ Rook path """
            raise MoveNotPossibleException

        if abs(event.start_pos[0] - event.end_pos[0]) != abs(event.start_pos[1] - event.end_pos[1]):
            """ Bishop path """
            raise MoveNotPossibleException

        """ Trace route """

        cols = list(range(event.start_pos[0], event.end_pos[0]))
        rows = list(range(event.start_pos[1], event.end_pos[1]))

        print(cols, rows)

        if len(rows) == 0:
            rows += [event.start_pos[1]] * len(cols)
        else:
            cols += [event.start_pos[0]] * len(rows)

        for pos in list(zip(cols, rows))[1:]:
            if board.get_piece(pos) is not None:
                raise MoveNotPossibleException

    def __str__(self):
        return 'Q'


class Event:
    NOTATION_REGEX = re.compile(
        r'(?P<piece>[KQRNB]?)(?P<start_pos>[a-h][1-8])?(?P<action>[-x])?(?P<end_pos>[a-h][1-8])(?P<event>(?=\+)|(?=#)|(?=\+\+)|(?=\(=\)))?'
    )

    MOVE = 'move'
    EAT = 'eat'

    CHECK = 'check'
    MATE = 'mate'
    EQUALITY = 'equality'

    def __init__(self,
                 piece: type[Piece],
                 start_pos: Optional[(int, int)],
                 end_pos: (int, int),
                 action: Union[Event.MOVE, Event.EAT],
                 event: Optional[Union[Event.CHECK, Event.MATE, Event.EQUALITY]]):
        self.piece = piece
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.action = action
        self.event = event

    def __str__(self):
        return {
            'piece': self.piece.__name__,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'action': self.action,
            'event': self.event,
        }.__str__()

    @classmethod
    def parse_event(cls, event: str):
        ev = cls.NOTATION_REGEX.match(event)

        piece: type[Piece]

        if ev.group('piece') == '':
            piece = Pawn
        elif ev.group('piece') == 'K':
            piece = King
        elif ev.group('piece') == 'Q':
            piece = Queen
        elif ev.group('piece') == 'R':
            piece = Rook
        elif ev.group('piece') == 'N':
            piece = Knight
        elif ev.group('piece') == 'B':
            piece = Bishop
        else:
            raise ValueError('invalid piece')

        start_position = None

        if ev.group('start_pos') is not None:
            start_position = (l2i(ev.group('start_pos')[0]), int(ev.group('start_pos')[1]) - 1)

        if ev.group('action') == '-' or ev.group('action') is None and start_position is None:
            action = Event.MOVE
        elif ev.group('action') == 'x':
            action = Event.EAT
        else:
            raise ValueError('invalid move type')

        end_position = (l2i(ev.group('end_pos')[0]), int(ev.group('end_pos')[1]) - 1)

        if ev.group('event') is None:
            event = None
        elif ev.group('event') == '+':
            event = Event.CHECK
        elif ev.group('event') == '#' or ev.group('event') == '++':
            event = Event.MATE
        elif ev.group('event') == '(=)':
            event = Event.EQUALITY
        else:
            raise ValueError('invalid event type')

        return cls(piece, start_position, end_position, action, event)


class Chess:
    def __init__(self):
        self.cells: list[list[Optional[Piece]]] = list()
        self.turn = PieceType.WHITE
        self._initialize()

    def _initialize(self):
        raise NotImplemented

    def handle_event(self, event: Event):
        if not isinstance(self.cells[event.start_pos[1]][event.start_pos[0]], event.piece):
            raise MoveNotPossibleException

        assert self.cells[event.start_pos[1]][event.start_pos[0]].color == self.turn
        self.cells[event.start_pos[1]][event.start_pos[0]].handle_event(self, event)
        self.turn = self.turn.opposite()

    def place_piece(self, piece: Piece, pos: (int, int)):
        start = self.find_position(piece)
        self.cells[start[1]][start[0]] = None
        self.cells[pos[1]][pos[0]] = piece

    def get_piece(self, pos: (int, int)) -> Optional[Piece]:
        return self.cells[pos[1]][pos[0]]

    def find_position(self, piece: Piece):
        for index, row in enumerate(self.cells):
            try:
                return row.index(piece), index
            except ValueError:
                pass

        raise ValueError('piece not found')


def l2i(letter: str):
    """ Converts board letter to list index. For example column 'A' will be 0 and 'H' will be 7 """
    letter = letter[0].upper()  # force letter to be a single character
    if letter < 'A' or letter > 'H':
        raise InvalidCellException

    return ord(letter) - 65


class ClassicChess(Chess):
    def _initialize(self):
        self.cells: list[list[Optional[Piece]]] = [
            [Rook(PieceType.WHITE), Knight(PieceType.WHITE), Bishop(PieceType.WHITE), Queen(PieceType.WHITE),
             King(PieceType.WHITE), Bishop(PieceType.WHITE), Knight(PieceType.WHITE), Rook(PieceType.WHITE)],
            [Pawn(PieceType.WHITE) for _ in range(8)],
            [None for _ in range(8)],
            [None for _ in range(8)],
            [None for _ in range(8)],
            [None for _ in range(8)],
            [Pawn(PieceType.BLACK) for _ in range(8)],
            [Rook(PieceType.BLACK), Knight(PieceType.BLACK), Bishop(PieceType.BLACK), Queen(PieceType.BLACK),
             King(PieceType.BLACK), Bishop(PieceType.BLACK), Knight(PieceType.BLACK), Rook(PieceType.BLACK)],
        ]

    def __str__(self):
        return str([
            [str(p) if p is not None else 'N' for p in row] for row in reversed(self.cells)
        ]).replace("[", "\n[")[3:-1]


class FisherChess(Chess):
    def _initialize(self):
        # Сама сделай генератор доски для шахмат Фишера :)
        # https://ru.wikipedia.org/wiki/%D0%A8%D0%B0%D1%85%D0%BC%D0%B0%D1%82%D1%8B-960
        pass


if __name__ == '__main__':
    chess = ClassicChess()

    while True:
        print(chess)
        chess.handle_event(Event.parse_event(input(f'Enter move {chess.turn}: ')))
