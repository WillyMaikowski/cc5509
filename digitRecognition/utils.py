def hasBlackLeft( img, i, j ):
    if j <= 0:
        return False
    for k in range( 0, j ):
        if img[i, k] == 0:  # es negro
            return True
    return False

def hasBlackRight( img, i, j ):
    if j >= len( img ) - 1:
        return False
    for k in range( j + 1, len( img ) ):
        if img[i, k] == 0:  # es negro
            return True
    return False

def hasBlackTop( img, i, j ):
    if i <= 0:
        return False
    for k in range( 0, i ):
        if img[k, j] == 0:  # es negro
            return True
    return False

def hasBlackBottom( img, i, j ):
    if i >= len( img ) - 1:
        return False
    for k in range( i + 1, len( img[0] ) ):
        if img[k, j] == 0:  # es negro
            return True
    return False

def hasBlackTopLeft( img, i, j ):
    if j <= 0 or i <= 0:
        return False
    while i > 0 and j > 0:
        i = i - 1
        j = j - 1
        if img[i, j] == 0:  # es negro
            return True
    return False

def hasBlackTopRight( img, i, j ):
    if j >= len( img ) - 1 or i <= 0:
        return False
    while i > 0 and j < len( img ) - 1:
        i = i - 1
        j = j + 1
        if img[i, j] == 0:  # es negro
            return True
    return False

def hasBlackBottomLeft( img, i, j ):
    if j <= 0 or i >= len( img ) - 1:
        return False
    while i < len( img ) - 1 and j > 0:
        i = i + 1
        j = j - 1
        if img[i, j] == 0:  # es negro
            return True
    return False

def hasBlackBottomRight( img, i, j ):
    if j >= len( img ) - 1 or i >= len( img ) - 1:
        return False
    while i < len( img ) - 1 and j < len( img ) - 1:
        i = i + 1
        j = j + 1
        if img[i, j] == 0:  # es negro
            return True
    return False

def m13C( img, i, j ):
    right = hasBlackRight( img, i, j )  # 1
    left = hasBlackLeft( img, i, j )  # 3
    top = hasBlackTop( img, i, j )  # 0
    bottom = hasBlackBottom( img, i, j )  # 2
    num_negros = 0
    if right:
        num_negros = num_negros + 1
    if left:
        num_negros = num_negros + 1
    if top:
        num_negros = num_negros + 1
    if bottom:
        num_negros = num_negros + 1

    if num_negros <= 1:
        return -1

    if num_negros == 2:
        if not top and not right:
            return 0
        if not right and not bottom:
            return 1
        if not bottom and not left:
            return 2
        if not left and not top:
            return 3

    if num_negros == 3:
        if not top:
            return 4
        if not right:
            return 5
        if not bottom:
            return 6
        if not left:
            return 7
    # num_negros == 4 =>encontrar direccion
    # Faltan los ultimos cuatro bins

def m4CC( img, i, j ):
    right = hasBlackRight( img, i, j )  
    left = hasBlackLeft( img, i, j )  
    top = hasBlackTop( img, i, j )
    bottom = hasBlackBottom( img, i, j )
    
    return int(top)<<3|int(right)<<2|int(bottom)<<1|int(left)

def m8CC( img, i, j ):
    right = hasBlackRight( img, i, j )  
    left = hasBlackLeft( img, i, j )  
    top = hasBlackTop( img, i, j )
    bottom = hasBlackBottom( img, i, j )
    topLeft = hasBlackTopLeft( img, i, j )
    bottomLeft = hasBlackBottomLeft( img, i, j )
    bottomRight = hasBlackBottomRight( img, i, j )
    topRight = hasBlackTopRight( img, i, j )
    
    return int(top)<<7|int(topRight)<<6|int(right)<<5|int(bottomRight)<<4|int(bottom)<<3|int(bottomLeft)<<2|int(left)<<1|int(topLeft)