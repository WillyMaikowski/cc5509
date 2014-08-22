
#Constantes
BACKGROUND = 240
FOREGROUND = 0
MARKED = 1

def mBlackTop( img ):
    aux = img.copy()
    for i in range( 1, len( aux ) ):
        for j in range( len( aux[i] ) ):
            if aux[i][j] == FOREGROUND:
                continue
            elif aux[i][j] == BACKGROUND and ( aux[i - 1][j] == FOREGROUND or aux[i - 1][j] == MARKED ):
                aux[i][j] = MARKED
    return aux

def mBlackLeft( img ):
    aux = img.copy()
    for i in range( 1, len( aux ) ):
        for j in range( len( aux[i] ) ):
            if aux[j][i] == FOREGROUND:
                continue
            elif aux[j][i] == BACKGROUND and ( aux[j][i - 1] == FOREGROUND or aux[j][i - 1] == MARKED ):
                aux[j][i] = MARKED
    return aux

def mBlackBottom( img ):
    aux = img.copy()
    for i in range( len( aux ) - 2, -1, -1 ):
        for j in range( len( aux[i] ) ):
            if aux[i][j] == FOREGROUND:
                continue
            elif aux[i][j] == BACKGROUND and ( aux[i + 1][j] == FOREGROUND or aux[i + 1][j] == MARKED ):
                aux[i][j] = MARKED
    return aux

def mBlackRight( img ):
    aux = img.copy()
    for i in range( len( aux ) - 2, -1, -1 ):
        for j in range( len( aux[i] ) ):
            if aux[j][i] == FOREGROUND:#negro
                continue
            elif aux[j][i] == BACKGROUND and ( aux[j][i + 1] == FOREGROUND or aux[j][i + 1] == MARKED ):#si pixel es blanco y derecha es negro o derecha marcado => marcar
                aux[j][i] = MARKED
    return aux

def mBlackTopLeft( img ):
    aux = img.copy()
    for i in range( len( aux ) ):
        for j in range( len( aux[i] ) ):
            if i-1 < 0 or j-1 < 0 or aux[i][j] == FOREGROUND:
                continue             
            elif aux[i][j] == BACKGROUND and ( aux[i - 1][j - 1] == FOREGROUND or aux[i - 1][j - 1] == MARKED ):
                aux[i][j] = MARKED
    return aux

def mBlackTopRight( img ):
    aux = img.copy()
    for i in range( len( aux ) ):
        for j in range( len( aux[i] ) ):
            if i-1 < 0 or j+1 >= len(aux[i]) or aux[i][j] == FOREGROUND:
                continue             
            elif aux[i][j] == BACKGROUND and ( aux[i - 1][j + 1] == FOREGROUND or aux[i - 1][j + 1] == MARKED ):
                aux[i][j] = MARKED
    return aux

def mBlackBottomLeft( img ):
    aux = img.copy()
    for i in range( len( aux )-2, -1, -1 ):
        for j in range( len( aux[i] ) ):
            if i+1>= len(img) or j-1 < 0 or aux[i][j] == FOREGROUND:
                continue             
            elif aux[i][j] == BACKGROUND and ( aux[i + 1][j - 1] == FOREGROUND or aux[i + 1][j - 1] == MARKED ):
                aux[i][j] = MARKED
    return aux

def mBlackBottomRight( img ):
    aux = img.copy()
    for i in range( len( aux )-2, -1, -1 ):
        for j in range( len( aux[i] ) ):
            if i+1>= len(img) or j+1 >= len(img[i]) or aux[i][j] == FOREGROUND:
                continue             
            elif aux[i][j] == BACKGROUND and ( aux[i + 1][j + 1] == FOREGROUND or aux[i + 1][j + 1] == MARKED ):
                aux[i][j] = MARKED
    return aux

def apply4CCv2( img ):
    top = mBlackTop( img )
    right = mBlackRight( img )
    bottom = mBlackBottom( img )
    left = mBlackLeft( img )
    aux = img.copy()
    for i in range( len( img ) ):
        for j in range( len( img[i] ) ):
            if aux[i][j] == FOREGROUND:
                aux[i][j] = -1
            else:
                aux[i][j] = ( int( top[i][j] ) << 3 | int( right[i][j] ) << 2 | int( bottom[i][j] ) << 1 | int( left[i][j] ) ) & 15
    return aux

def apply8CCv2( img ):
    topRight = mBlackTopRight( img )
    topLeft = mBlackTopLeft( img )
    bottomRight = mBlackBottomRight( img )
    bottomLeft = mBlackBottomLeft( img )
    aux = img.copy()
    for i in range( len( img ) ):
        for j in range( len( img[i] ) ):
            if aux[i, j] == FOREGROUND:
                aux[i, j] = -1
            else:
                aux[i, j] = ( int( topRight[i][j] ) << 3 | int( topLeft[i][j] ) << 2 | int( bottomRight[i][j] ) << 1 | int( bottomLeft[i][j] ) ) & 15
    return aux

def apply13Cv2( img ):
    top = mBlackTop( img )
    right = mBlackRight( img )
    bottom = mBlackBottom( img )
    left = mBlackLeft( img )
    aux = img.copy()
    for i in range( len( aux ) ):
        for j in range( len( aux[i] ) ):
            if aux[i, j] == FOREGROUND:
                aux[i, j] = -1
            else:
                aux[i, j] = m13Cv2( right, left, top, bottom, i, j )
    return aux


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

# retorna true si hay una salida por s1
def hasS1( img, i, j ):
    if i <= 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i - 1, -1, -1 ):
        if img[k, j] == 0:  # negro -> dejar de buscar
            return False
        if not hasBlackLeft( img, k, j ):
            return True
    return False

# retorna true si hay una salida por s1
def hasS1v2( imgTL, i, j ):
    if i <= 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i - 1, -1, -1 ):
        if imgTL[k, j] == FOREGROUND:  # negro -> dejar de buscar
            return False
        if not imgTL[k,j]==MARKED:
            return True
    return False

# retorna true si hay una salida por s2
def hasS2( img, i, j ):
    if i <= 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i - 1, -1, -1 ):
        if img[k, j] == 0:  # negro -> dejar de buscar
            return False
        if not hasBlackRight( img, k, j ):
            return True
    return False

# retorna true si hay una salida por s2
def hasS2v2( imgTR, i, j ):
    if i <= 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i - 1, -1, -1 ):
        if imgTR[k, j] == FOREGROUND:  # negro -> dejar de buscar
            return False
        if not imgTR[k,j]==MARKED:
            return True
    return False

# retorna true si hay una salida por s3
def hasS3( img, i, j ):
    if i >= len( img ) - 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i + 1, len( img ) ):
        if img[k, j] == 0:  # negro -> dejar de buscar
            return False
        if not hasBlackLeft( img, k, j ):
            return True
    return False

# retorna true si hay una salida por s3
def hasS3v2( imgBL, i, j ):
    if i >= len( imgBL ) - 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i + 1, len( imgBL ) ):
        if imgBL[k, j] == FOREGROUND:  # negro -> dejar de buscar
            return False
        if not imgBL[k,j]==MARKED:
            return True
    return False

# retorna true si hay una salida por s4
def hasS4( img, i, j ):
    if i >= len( img ) - 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i + 1, len( img ) ):
        if img[k, j] == 0:  # negro -> dejar de buscar
            return False
        if not hasBlackRight( img, k, j ):
            return True
    return False

# retorna true si hay una salida por s4
def hasS4v2( imgBR, i, j ):
    if i >= len( imgBR ) - 1:
        return False
    # se asume que el pixel esta rodeado en toda direccion
    for k in range( i + 1, len( imgBR ) ):
        if imgBR[k, j] == FOREGROUND:  # negro -> dejar de buscar
            return False
        if not imgBR[k,j]==MARKED:
            return True
    return False


def apply4CC( img ):
    aux = img.copy()
    for i in range( len( img[0] ) ):
        for j in range( len( img ) ):
            if img[i, j] == 0:
                aux[i, j] = -1
            else:
                aux[i, j] = m4CC( img, i, j )
    return aux

def apply8CC( img ):
    aux = img.copy()
    for i in range( len( img[0] ) ):
        for j in range( len( img ) ):
            if img[i, j] == 0:
                aux[i, j] = -1
            else:
                aux[i, j] = m8CC( img, i, j )
    return aux

def apply13C( img ):
    aux = img.copy()
    for i in range( len( img[0] ) ):
        for j in range( len( img ) ):
            if img[i, j] == 0:
                aux[i, j] = -1
            else:
                aux[i, j] = m13C( img, i, j )
    return aux

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
    if not hasS1:
        return 9

    if not hasS2:
        return 10

    if not hasS3:
        return 11

    if not hasS4:
        return 12

    return 8  # punto interior

def m13Cv2( auxRight, auxLeft, auxTop, auxBottom, i, j ):
    right = auxRight[i,j]==MARKED  # 1
    left = auxLeft[i,j]==MARKED  # 3
    top = auxTop[i,j]==MARKED  # 0
    bottom = auxBottom[i,j]==MARKED  # 2
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
    if not hasS1v2(auxLeft,i,j):
        return 9

    if not hasS2v2(auxRight,i,j):
        return 10

    if not hasS3v2(auxLeft,i,j):
        return 11

    if not hasS4v2(auxRight,i,j):
        return 12

    return 8  # punto interior

def m4CC( img, i, j ):
    right = hasBlackRight( img, i, j )
    left = hasBlackLeft( img, i, j )
    top = hasBlackTop( img, i, j )
    bottom = hasBlackBottom( img, i, j )

    return int( top ) << 3 | int( right ) << 2 | int( bottom ) << 1 | int( left )

def m8CC( img, i, j ):
    topLeft = hasBlackTopLeft( img, i, j )
    bottomLeft = hasBlackBottomLeft( img, i, j )
    bottomRight = hasBlackBottomRight( img, i, j )
    topRight = hasBlackTopRight( img, i, j )

    return int( topRight ) << 3 | int( topLeft ) << 2 | int( bottomRight ) << 1 | int( bottomLeft )