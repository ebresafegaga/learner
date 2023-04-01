#lang racket/base

(require malt)

(define •-
  (λ (w t)
    (sum (* w t))))

(define • dot-product)

(define plane-xs
  (tensor (tensor 1.0 2.05)
          (tensor 1.0 3.0)
          (tensor 2.0 2.0)
          (tensor 2.0 3.91)
          (tensor 3.0 6.13)
          (tensor 4.0 8.09)))

(define plane-ys
  (tensor 13.99
          15.99
          18.0
          22.4
          30.4
          37.94))

(define line-xs
  (tensor 2.0 1.0 4.0 3.0))

(define line-ys
  (tensor 1.8 1.2 4.2 3.3))

(define quad-xs [tensor -1.0 0.0 1.0 2.0 3.0])

(define quad-ys [tensor 2.55 2.1 4.35 10.2 18.25])

(define samples
  (λ (n s)
    (sampled n s (list))))

(define sampled
  (λ (n i a)
    (cond
      [(zero? i) a]
      [else
       (sampled n (sub1 i)
                (cons (random n) a))])))

(define line
  (λ (x)
    (λ (θ)
      (+ (* (ref θ 0) x) (ref θ 1)))))


(define quad
  (λ (t)
    (λ (θ)
      (+ (* (ref θ 0) (sqr t))
         (+ (* (ref θ 1) t) (ref θ 2))))))

(define plane
  (λ (t)
    (λ (θ)
      (+ (• (ref θ 0) t)
         (ref θ 1)))))

(define l2-loss
  (λ (target)
    (λ (xs ys)
      (λ (θ)
        (let ([pred-ys ((target xs) θ)])
          (sum
           (sqr
            (- ys pred-ys))))))))

(declare-hyper batch-size)

(define sampling-obj
  (λ (expectant xs ys)
    (let ([n (tlen xs)])
      (λ (θ)
        (let ([b (samples n batch-size)])
          ((expectant (trefs xs b) (trefs ys b)) θ))))))

(define revise
  (λ (f revs θ)
    (cond
      [(zero? revs) θ]
      [else (revise f (sub1 revs) (f θ))])))

(define obj ((l2-loss line) line-xs line-ys))

;;(let ([α 0.01]
;;      [obj ((l2-loss line) line-xs line-ys)])
;  (let ([f (λ (θ)
;             (let ([gs (∇ obj θ)])
;               (map (λ (p g) (- p (* α g)))
;                    θ
;                    gs) ))])
;    (revise f 1000 (list 0.0 0.0))))

;; ((quad 3.0) (list 4.5 2.1 7.8))

;((plane (tensor 2.0 3.91))
;  (list
;   (tensor 3.999101424334315 1.9865038609052141)
;   6.111996347039005))

; (trefs [tensor 5.0 2.8 4.2 2.3 7.4 1.7 8.1] (list 6 0 3 1))


(declare-hyper α)

(define lonely-i
  (λ (p)
    (list p)))

(define lonely-d
  (λ (P)
    (ref P 0)))

(define lonely-u
  (λ (P g)
    (list (- (ref P 0) (* α g)))))

(define naked-i
  (λ (p)
    (let ([P p])
      P)))

(define naked-d
  (λ (P)
    (let ([p P])
      p)))

(define naked-u
  (λ (P g)
    (- P (* α g))))

(define gradient-descent
    (λ (obj θ)
      (let ([f (λ (Θ)
                 (map (λ (p g)
                        (- p (* α g)))
                      Θ
                      (∇ obj Θ)))])
        (revise f revs θ))))

(define grate-gradient-descent
  (λ (inflate deflate update)
    (λ (obj θ)
      (let ([f (λ (Θ)
                 (map update
                      Θ
                  (∇ obj
                     (map deflate Θ))))])
        (map deflate
             (revise f revs
                     (map inflate θ)))))))

(define lonely-gradient-descent
  (grate-gradient-descent lonely-i lonely-d lonely-u))

(define naked-gradient-descent
  (grate-gradient-descent naked-i naked-d naked-u))

(define try-plane
  (λ (a-gradient-descent a-revs an-a)
    (with-hypers
        ([revs a-revs]
         [α an-a]
         [batch-size 4])
      (a-gradient-descent
       (sampling-obj
        (l2-loss plane)
        plane-xs
        plane-ys)
       (list (tensor 0.0 0.0)
             0.0)))))

(declare-hyper μ)

(define Z
  (λ (shape)
    (cond
      [(null? shape) 0]
      [else
       (let* ([size (car shape)]
              [ts (build-list size (λ (x) x))])
         (apply tensor (for/list ([t ts])
                         (Z (cdr shape)))))])))

(define zeros
  (λ (p)
    (Z (shape p))))

(define velocity-i
  (λ (p)
    (list p (zeros p))))

(define velocity-d
  (λ (P)
    (ref P 0)))

(define velocity-u
  (λ (P g)
    (let ([v (- (* μ (ref P 1)) (* α g))])
      (list (+ (ref P 0) v) v))))

(define velocity-gradient-descent
  (grate-gradient-descent velocity-i velocity-d velocity-u ))

;(with-hypers
;    ([μ 0.9])
;  (try-plane velocity-gradient-descent 5000))


(define ϵ 1e-08)
(declare-hyper β)

(define (trace r)
  (print r)
  r)

(define tensor->list
  (λ (t)
    (cond
      [(number? t) t]
      [else (vector->list (vector-ref t 2))])))

(define smooth
  (λ (decay-rate average g)
    (+ (* decay-rate average)
       (* (- 1.0 decay-rate) g))))

(define rms-u
  (λ (P g)
    (let* ([r (smooth β (ref P 1) (sqr g))]
           [α-hat (/ α (+ (sqrt r) ϵ))]) 
      (list (- (ref P 0) (* α-hat g)) r)))) 

(define rms-i
  (λ (p)
    (list p (zeroes p))))

(define rms-d
  (λ (P)
    (ref P 0)))

(define rms-gradient-descent
  (grate-gradient-descent rms-i rms-d rms-u))


(define adam-u
  (λ (P g)
    (let* ([r (smooth β (ref P 2) (sqr g))]
           [α-hat (/ α (+ (sqrt r) ϵ))]
           [v (smooth μ (ref P 1) g)])
      (list (- (ref P 0) (* α-hat v)) v r))))

(define adam-i
  (λ (p)
    (let* ([v (zeros p)]
           [r v])
      (list p v r))))

(define adam-d
  (λ (P)
    (ref P 0)))

(define adam-gradient-descent
  (grate-gradient-descent adam-i adam-d adam-u))

(with-hypers
    ([μ 0.85]
     [β 0.9])
  (try-plane adam-gradient-descent 1500 0.01))

;(with-hypers
;    ([β 0.9])
;  (try-plane rms-gradient-descent 3000 0.01))



;; (try-plane naked-gradient-descent)

; '(1.0499993623489503 1.8747718457656533e-6)

;(with-hypers
;    ([revs 1000]
;     [α 0.01]
;     [batch-size 4])
;  (gradient-descent
;   ;obj
;   (sampling-obj (l2-loss line)
;                 line-xs
;                 line-ys)
;   (list 0.0 0.0)))


; (• [tensor 2.0 1.0 7.0] [tensor 8.0 4.0 3.0])

;(with-hypers
;    ([revs 1000]
;     [α 0.001])
;  (gradient-descent
;   ((l2-loss plane) plane-xs plane-ys)
;   (list (tensor 0.0 0.0) 0.0)))


;;(with-hypers
;    ([revs 1000]
;     [α 0.001])
;  (gradient-descent
;   ((l2-loss quad) quad-xs quad-ys)
;   (list 0.0 0.0 0.0)))

;(define f
;  (λ (θ)
;    (map (λ (p)
;           (- p 3))
;         θ)))


;; (revise f 5 (list 1 2 3))

;; (∇ obj (list 0.0 0.0))

;; (∇ (λ (θ) (sqr (ref θ 0))) (list 27))

;; (((l2-loss line) line-xs line-ys) (list 62.63 0))


;; ((line line-xs) (tensor 0.0 0.0))


(define π 3.141592653589793)
(define tensor-example [tensor 5.0 7.18 π])
(define tensor-example-length (tlen tensor-example))


(define tensor-rank
  (λ (t)
    (cond
      [(scalar? t) 0]
      [else (add1 (tensor-rank (tref t 0)))])))

;; [[5.2 6.3 8.0] [6.9 7.1 0.5]] => (list 2 3)

(define tensor-shape
  (λ (t)
    (cond
      [(scalar? t) '()]
      [else (cons (tlen t) (tensor-shape (tref t 0)))])))


;; (sum (tensor 10.0 12.0 14.0))

;;(sqrt (tensor 4 16 25))

;; (+ 2 (tensor 1 2 3))

;; (+ (tensor 2) (tensor 7))


;; ((line 10) (tensor 8 10))
