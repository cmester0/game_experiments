var c = document.getElementById("myCanvas");
var ctx = c.getContext("2d");

var width = 1920 - 50;
var height = 1080 - 50;

// Steering Behavior
// Behavior Trees

c.style.cursor = "none";

function col (a,b) {
    if ((b[0] <= a[0]+a[2] && a[0]+a[2] <= b[0] + b[2]) ||
	(b[0] <= a[0]      && a[0]      <= b[0] + b[2]) ||
	(b[0] <= a[0] && a[0] + a[2] <= b[0] + b[2]) ||
	(a[0] <= b[0] && b[0] + b[2] <= a[0] + a[2])) {
	if ((b[1] <= a[1]+a[3] && a[1]+a[3] <= b[1] + b[3]) ||
	    (b[1] <= a[1]      && a[1]      <= b[1] + b[3]) ||
	    (b[1] <= a[1] && a[1] + a[3] <= b[1] + b[3]) ||
	    (a[1] <= b[1] && b[1] + b[3] <= a[1] + a[3])) {
	    return true;
	}
    }
    return false;
}

let Vector = class {
    constructor(x, y) {
	this.x = x;
	this.y = y;
    }

    size() {
	return Math.sqrt(this.x ** 2 + this.y ** 2);
    }

    truncate(value) {
	if (this.size() < value) {
	    return this;
	}
	else {
	    return this.normalize().scale(value);
	}
    }

    scale(n) {
	return new Vector(this.x * n, this.y * n);
    }

    scale_vec(n) {
	return new Vector(this.x * n.x, this.y * n.y);
    }
    
    normalize () {
	if (this.size() > 0) {
	    return this.scale(1.0 / this.size());
	} else {
	    return new Vector(0,0);
	}
    }

    add (v) {
	return new Vector(this.x + v.x, this.y + v.y);
    }

    sub (v) {
	return new Vector(this.x - v.x, this.y - v.y);
    }

    set_angle(value) {
	return new Vector(Math.cos(value), Math.sin(value)).scale(this.size());
    }

};

var group_a = [{p: new Vector(100, 100), v: new Vector(0.0, 0.0), vel: 20},
	       {p: new Vector(100, 100), v: new Vector(0.0, 0.0), vel: 10},
	       {p: new Vector(100, 100), v: new Vector(0.0, 0.0), vel: 5},
	       {p: new Vector(100, 100), v: new Vector(0.0, 0.0), vel: 1}];
var hunt = {p: new Vector(800, 320), v: new Vector(0.0, 0.0), vel: 10};
var danger = {p: new Vector(800, 320), v: new Vector(0.0, 0.0), vel: 10};

var t = 0;
var dt = 0.15;

var count = 0;

function update_pos_vel(pos, vel) {
    let p = pos.add(vel.scale(dt));
    let v = vel;
    
    if (p.x < 0) {
	p.x = -p.x;
	v = new Vector(Math.abs(v.x), v.y);
    }
    if (p.x > width) {
	p.x = 2 * width - p.x;
	v = new Vector(-Math.abs(v.x), v.y);
    }
    if (p.y < 0) {
	p.y = -p.y;
	v = new Vector(v.x, Math.abs(v.y));
    }
    if (p.y > height) {
	p.y = 2 * height - p.y;
	v = new Vector(v.x, -Math.abs(v.y));
    }
    
    return {pos: p, vel: v};
}

function circular_move(obj) {
    let a = new Vector(800 - obj.p.x, 520 - obj.p.y);
    a = a.normalize();
    a = a.scale(0.2);
    
    let v = obj.v.add(a);
    
    let pv = update_pos_vel(obj.p, v);
    let p = pv.pos;
    v = pv.vel;
     
    return {p: p, v: v};
}

function right_move(obj) {
    let v = new Vector(1, 1).normalize().scale(obj.vel);

    let pv = update_pos_vel(obj.p, v);
    let p = pv.pos;
    v = pv.vel;
    
    if (p.y > height) {
	p.y = 0;
    }
    if (p.x > width) {
	p.x = 0;
    }
    
    return {p: p, v: v};
}

function random(from, size) {
    return from.add(new Vector(Math.random(), Math.random()).scale_vec(size));
}

function seek(obj, target) {
    let max_force = 1.0;
    return target.p.sub(obj.p).normalize().scale(max_force);
}

function arrive(obj, target) {
    let desired = target.p.sub(obj.p);
    let distance = desired.size();

    let slowRadius = 100.0;
    
    if (distance < slowRadius) {
	desired = desired.normalize().scale(obj.vel).scale(distance / slowRadius);
    } else {
	desired = desired.normalize().scale(obj.vel)
    }
    
    return desired.sub(obj.v);
}

function flee(obj, target) {
    let desired = target.p.sub(obj.p).scale(-1);
    let distance = desired.size();

    let slowRadius = 100.0;
    
    if (distance < slowRadius) {
	desired = desired.normalize().scale(obj.vel).scale(distance / slowRadius);
    } else {
	desired = desired.normalize().scale(obj.vel)
    }
    
    return desired.sub(obj.v);
}
 
var wanderAngles = [0, 1, 2, 3];
function wander(obj, wanderAngle) {
    var v = obj.v.set_angle(wanderAngle);

    let pv = update_pos_vel(obj.p, v);
    let p = pv.pos;
    v = pv.vel;

    let wa = wanderAngle - 0.05 + Math.random() * 0.1;
    
    return {p: p, v: v, wander: wa};
}

function persue(obj, target) {
    let T = obj.p.sub(target.p).size() / obj.vel;
    let t = {...target, p: target.p.add(target.v.scale(T))};
    
    return seek(obj, t);
}

function evade(obj, target) {
    let T = obj.p.sub(target.p).size() / obj.vel;
    let t = {...target, p: target.p.add(target.v.scale(T))};
    
    return flee(obj, t);
}

function initialize () {
    for (var i = 0; i < group_a.length; i++) {
	group_a[i].v = new Vector(group_a[i].vel, 0);
    }

    hunt.p = random(new Vector(0,0), new Vector(width, height));
    range = 10;
    hunt.v = random(new Vector(-range,-range), new Vector(2 * range, 2 * range));

    danger.p = random(new Vector(0,0), new Vector(width, height));
    range = 10;
    danger.v = random(new Vector(-range,-range), new Vector(2 * range, 2 * range));
}

function update_steering (steering, obj) {
    obj.v = obj.v.add(steering).truncate(obj.vel);

    let pv = update_pos_vel(obj.p, obj.v);
    obj.p = pv.pos;
    obj.v = pv.vel;

    return obj;
}

function update () {
    t += dt;
    count += 1;
    
    // var target_pos = right_move(target); // circular_move(target);
    // target.p = target_pos.p;
    // target.v = target_pos.v;

    let pv = update_pos_vel(hunt.p, hunt.v);
    hunt.p = pv.pos;
    hunt.v = pv.vel;

    pv = update_pos_vel(danger.p, danger.v);
    danger.p = pv.pos;
    danger.v = pv.vel;
    
    for (var i = 0; i < group_a.length; i++) {
	var steering = new Vector(0,0);
	steering = steering.add(evade(group_a[i], danger).scale(1));
	steering = steering.add(persue(group_a[i], hunt).scale(1));
	group_a[i] = update_steering(steering, group_a[i]);
	
	// if (group_a[i].p.sub(target.p).size() < 400) {
	//     var steering = flee(group_a[i], target);
	// } else {
	//     var steering = wander(group_a[i], wanderAngles[i]);
	//     wanderAngles[i] = steering.wander;
	// }
	
	// group_a[i].p = steering.p;
	// group_a[i].v = steering.v;
    }
}

function draw () {
    ctx.fillStyle = "#00FF00";    
    for (var i = 0; i < group_a.length; i++) {
	ctx.fillRect(group_a[i].p.x, group_a[i].p.y, 50, 50);
    }
     
    ctx.fillStyle = "#FF0000";
    ctx.fillRect(danger.p.x, danger.p.y, 50, 50);
     
    ctx.fillStyle = "#0000FF";
    ctx.fillRect(hunt.p.x, hunt.p.y, 50, 50);
    
    // ctx.fillStyle = "#000000";
    // ctx.font = "40px Georgia";
    // ctx.fillText(timer[level_index],10,40);
}
    
function loop() {
    ctx.clearRect(0,0,width + 50,height + 50);
    draw();
    update();
}

initialize ();
setInterval(loop,10);
